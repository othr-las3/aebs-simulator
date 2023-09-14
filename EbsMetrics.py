"""Implementation of the causal mechanisms considered in the simulator.
"""
import numpy as np


def calc_a_brake_real(a_ideal, mu_real, mu_ideal=0.51):
    """Calculates the resulting (i.e. based on the sampled params.) decceleration of the ego.

    Args:
        a_ideal (float): Ideal (i.e., maximum) expected decceleration of the ego.
        mu_real (float): Resulting friction coefficient based on wheater.
        mu_ideal (float, optional): Ideal (i.e., maximum) friction coefficient under ideal conditions.

    Returns:
        float: Resulting (i.e. based on the sampled params.) decceleration of the ego.
    """
    mu_real = mu_real if mu_real > 0.0 else mu_ideal - 0.3
    mu_ideal = mu_ideal if mu_ideal > 0.0 else 0.51
    return a_ideal * (mu_real / mu_ideal)


def calc_friction_on_rain(rain_intensity, mu_min=0.21, mu_max=0.51):
    """Calculates the resulting (i.e. based on the sampled params.) friction of the ego.

    Args:
        rain_intensity (float): Current rain intensity
        mu_min (float, optional): Cut-off for minimum friction
        mu_max (float, optional): Cut-off for maximum friction

    Returns:
        float: Resulting (i.e. based on the sampled params.) friction of the ego.
    """
    spread = abs(mu_min - mu_max)
    radiant_itensity = np.pi * 0.5 * rain_intensity / 100
    interpol_val = np.sin(radiant_itensity)

    offset_val = spread * interpol_val
    real_val = mu_max - offset_val
    return real_val


def calc_x_first_detection(fog_density, rain_intensity, is_day, poly_degree=2):
    """Calculates the resulting (i.e. based on the sampled params.) distance of first detection.
        That is the clearance between ego and agent under the current environmental conditons that allows a successful classification of the agent.

    Args:
        fog_density (float): Current fog intensity
        rain_intensity (float): Current rain intensity
        is_day (bool): Whether it is day or night
        poly_degree (int, optional): Which polynomial fitting should be used (1:linear, 2:quad., 3:cubic) to calculate the distance of first detection.

    Returns:
        float: Resulting (i.e. based on the sampled params.) distance of first detection.
    """
    # Polynomial Regression Degree = 2
    interpol_val = 0.0

    if poly_degree == 1:
        if is_day:
            factor_fog = -0.41198347 * fog_density
            factor_rain = -0.2161157 * rain_intensity
            intercept = 83.30578512396694
            interpol_val = factor_fog + factor_rain + intercept

        else:
            factor_fog = -0.22272727 * fog_density
            factor_rain = -0.39090909 * rain_intensity
            intercept = 72.5
            interpol_val = factor_fog + factor_rain + intercept

    if poly_degree == 2:
        if is_day:
            factor_fog = -1.09923183e00 * fog_density + 5.27124391e-03 * np.square(
                fog_density
            )
            factor_rain = -3.31208943e-01 * rain_intensity - 4.50307268e-04 * np.square(
                rain_intensity
            )
            compound_term = 3.20247934e-03 * fog_density * rain_intensity
            intercept = 98.5433884297531
            interpol_val = factor_fog + factor_rain + compound_term + intercept

        else:
            factor_fog = -5.18282475e-01 * fog_density + 1.64229710e-04 * np.square(
                fog_density
            )
            factor_rain = -9.49761602e-01 * rain_intensity + 2.79720280e-03 * np.square(
                rain_intensity
            )
            compound_term = 5.58264463e-03 * fog_density * rain_intensity
            intercept = 90.89876033057809

            interpol_val = factor_fog + factor_rain + compound_term + intercept

    if poly_degree == 3:
        if is_day:
            factor_fog = (
                -1.33554072e00 * fog_density
                + 1.11437805e-02 * np.square(fog_density)
                - 2.93141202e-05 * np.power(fog_density, 3)
            )
            factor_rain = (
                2.47156884e-01 * rain_intensity
                - 1.25105955e-02 * np.square(rain_intensity)
                + 6.28664265e-05 * np.power(rain_intensity, 3)
            )
            compound_term = (
                8.92667938e-04 * fog_density * rain_intensity
                - 2.95083704e-05 * np.square(fog_density) * rain_intensity
                + 5.26064844e-05 * fog_density * np.square(rain_intensity)
            )
            intercept = 95.60314685322214
            interpol_val = factor_fog + factor_rain + compound_term + intercept

        else:
            factor_fog = (
                -3.46952038e-01 * fog_density
                + -7.01419792e-03 * np.square(fog_density)
                + 5.57144875e-05 * np.power(fog_density, 3)
            )
            factor_rain = (
                -2.50568623e-01 * rain_intensity
                - 1.80599703e-02 * np.square(rain_intensity)
                + 1.45599350e-04 * np.power(rain_intensity, 3)
            )
            compound_term = (
                9.90559441e-03 * fog_density * rain_intensity
                - 2.35749099e-05 * np.square(fog_density) * rain_intensity
                - 1.96545878e-05 * fog_density * np.square(rain_intensity)
            )
            intercept = 86.8936745073907
            interpol_val = factor_fog + factor_rain + compound_term + intercept

    return interpol_val


def calc_x_activation(v_ego, v_agent, target_ttc):
    """Calculates the resulting (i.e. based on the sampled params.) EBS activation distance of the ego.
        That is TTC translated to a distance.

    Args:
        v_ego (float): Velocity of the ego.
        v_agent (float):  Velocity of the agent.
        target_ttc (float): Time to collision

    Returns:
        float: Resulting (i.e. based on the sampled params.) EBS activation distance of the ego.
    """
    v_rel = calc_rel_velocity(v_ego, v_agent)
    return (
        abs(v_rel * target_ttc) if v_rel < 0.0 else 333.3 + 50
    )  # ttc_max = 6sec * v_rel_max = 200km/h + offset # [calc_x_first_detection(fog_density=0, rain_intensity=0, is_day=True) + 50] #defaulting for Bayesserver clustering


def calc_x_braking(x_first_detection, x_activation):
    """Calculates the resulting (i.e. based on the sampled params.) distance at which all conditions for emergency breaking of the ego are fullfilled.

    Args:
        x_first_detection (float): Distance of first detection.
        x_activation (float): EBS activation distance of the ego.

    Returns:
        float: Resulting (i.e. based on the sampled params.) distance at which all conditions for emergency breaking of the ego are fullfilled.
    """
    return min(
        x_first_detection, x_activation
    )  # x_activation is distance defined by ttc value and relative velocity


## generic function for calculations
def calc_min_d_brake(v_ego, mu_road=0.51, a_brake=5.0):
    """Calculates minimum distance to deccelerate to 0m/s

    Args:
        v_ego (float): Velocity of the ego.
        mu_road (float, optional): Friction coefficient of the road/tire system.
        a_brake (float, optional): Decceleration capability of the ego.

    Returns:
        float: Minimum distance to deccelerate to 0m/s
    """
    return np.square(v_ego) / (2.0 * mu_road * np.abs(a_brake))


def calc_v_after_accelaration(v_init, a, t_delta):
    """Calculates the resulting velocity after an acceleration for a given time.

    Args:
        v_init (float): Initial velocity before acceleration.
        a (float): Acceleration
        t_delta (float): Time span of acceleration

    Returns:
        float: Resulting velocity after an acceleration for a given time.
    """
    return v_init + a * t_delta


def calc_dist_moved(v, a, t_delta):
    """Calculates the covered distance after an acceleration for a given time.

    Args:
        v (float): Initial velocity before acceleration.
        a (float): Acceleration
        t_delta (float): Time span of acceleration

    Returns:
        float: Covered distance after an acceleration for a given time.
    """
    return v * t_delta + 0.5 * a * np.square(t_delta)


def calc_rel_velocity(v_ego, v_agent):
    """Calculates relative velocity between ego and agent.

    Args:
        v_ego (float): Velocity of the ego.
        v_agent (float): Velocity of the agent.

    Returns:
        float: Relative velocity between ego and agent.
    """
    return (
        v_agent - v_ego
    )  # // if ego (vehicle behind) is faster --> v_rel is negative --> ego needs to brake


def calc_rel_acceleartion(a_ego, a_agent):
    """Calculates relative acceleration between ego and agent.

    Args:
        a_ego (float): Aacceleration of the ego.
        a_agent (float): Aacceleration of the agent.

    Returns:
        float: Relative acceleration between ego and agent.
    """
    return a_agent - a_ego


def calc_collision_energy(
    v_ego, v_agent, mass_ego=1900
):  # x_clearance, a_break=5.0, ):
    """Calculates the resulting collision energy between ego and agent.

    Args:
        v_ego (float): Velocity of the ego.
        v_agent (float): Velocity of the agent.
        mass_ego (int, optional): Mass of the agent
    """
    v_rel = calc_rel_velocity(v_ego, v_agent)
    return 0.5 * mass_ego * np.square(v_rel)


def is_ebs_active(x_clearance, x_activation, x_first_detection):
    """Evaluate if all conditions for an EBS activation are present.

    Args:
        x_clearance (float): Current clearance between ego and agent
        x_activation (float): EBS activation distance of the ego.
        x_first_detection (float): Distance of first detection.

    Returns:
        bool: Whether or not emergency breaking should be active.
    """
    x_braking = calc_x_braking(x_first_detection, x_activation)

    ebs_condition_fullfilled = False

    if x_clearance <= x_braking:
        ebs_condition_fullfilled = (
            x_clearance <= x_activation
        )  ##x_first_detection does not trigger braking -> only delays it

    return ebs_condition_fullfilled


# def calc_ttc(v_ego, v_agent, x_clearance):
# 	return -1.0 * x_clearance / (calc_rel_velocity(v_ego, v_agent) + 1e-10 )  #ttc when postivie indicates potential crash situation (faster ego)


# def calc_enhanced_time_to_collision(v_ego, v_agent, a_ego, a_agent, x_clearance):
# 	v_rel = calc_rel_velocity(v_ego, v_agent)
# 	a_rel = calc_rel_acceleartion(a_ego, a_agent)
#
# 	nominator = -1.0 * v_rel - np.sqrt( np.square(v_rel) - 2*a_rel*x_clearance )
#
# 	return nominator / a_rel


# def is_ebs_active(ttc, ttc_min, t_step, t_reaction, is_delayed=False, elapsed_delay=None):
# 	if ttc > ttc_min:
# 		return False

# 	if is_delayed:
# 		if elapsed_delay >= t_reaction:
# 			return True
# 		else:
# 			return False
