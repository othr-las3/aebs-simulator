"""Summary
"""
from EbsDistributions import *


class SimulatorConfig:

    """Parameterization of a car-follows-car scenario.

    Attributes:
        dstrb_a_brake (EbsDistribution): Distribution representing the range of values for ideal breaking decceleration.
        dstrb_clearance (EbsDistribution): Distribution representing the range of values for the initial clearance between ego and agent car.
        dstrb_fog_density (EbsDistribution): Distribution representing the range of values for fog (as implemented by CARLA).
        dstrb_is_day (EbsDistribution): Distribution representing the range of values for whether it is day (12am) or night (12pm).
        dstrb_rain_intensity (EbsDistribution): Distribution representing the range of values for rain (as implemented by CARLA).
        dstrb_ttc (EbsDistribution): Distribution representing the range of values for the time to collision (interpreted as SW parameterization of the ADAS)
        dstrb_v_agent (EbsDistribution): Distribution representing the range of values for the agents velocity at simulation start.
        dstrb_v_ego (EbsDistribution): Distribution representing the range of values for the ego velocity at simulation start.
        log_file_name (str): Name of the output file name for a single simulation.
        max_mu_fric (float): Upper limit of the friction coefficient.
        min_mu_fric (float): Lower limit of the friction coefficient.
        poly_degree_first_detection (int): Which polynomial fitting should be used (1:linear, 2:quad., 3:cubic).
        t_simulation (int): Total length of a simulation run.
        t_step (float): Delta time between a single simulation step and the next (t_1 =  t_step + t_0).
    """

    dstrb_v_ego = TruncNormal(
        lower_clip=60.0, upper_clip=200.0, mu=115.0, sigma=33.0
    )  # km/h
    dstrb_v_agent = TruncNormal(
        lower_clip=0.0, upper_clip=200.0, mu=115.0, sigma=33.0
    )  # km/h
    dstrb_clearance = Uniform(
        lower_clip=380.0, upper_clip=500.0
    )  # meter  max 90km/h difference @ 6sec TTC = 150m upperbound is not relevant
    dstrb_a_brake = TruncNormal(
        lower_clip=4.0, upper_clip=6.0, mu=5.0, sigma=1.0
    )  # m/s2 -> 0.51g according to spec

    # dstrb_mu_road 		= TruncNormal(lower_clip=0.21, upper_clip=0.51, mu=0.36, sigma=0.15)
    dstrb_ttc = TruncNormal(lower_clip=4.0, upper_clip=6.0, mu=5.0, sigma=1.0)  # sec
    dstrb_fog_density = TruncPareto(
        shape_factor=7.0, lower_clip=0.0, upper_clip=100.0
    )  # dimless
    dstrb_rain_intensity = TruncPareto(
        shape_factor=7.0, lower_clip=0.0, upper_clip=100.0
    )  # dimless
    dstrb_is_day = Binomial(prob=0.5139)  # binary, night = 1 - prob_is_day = 1-0.5139

    min_mu_fric = 0.28  # 0.21  # friction coefficient wet road
    max_mu_fric = 0.70  # 0.51  # friction coeefficient dry road

    t_simulation = 100  # sec.
    t_step = 0.1  # sec.

    poly_degree_first_detection = 3  # which fit to yolov3 data should be used
    log_file_name = "Param_log.csv"

    def __init__(self, **kwargs):
        """Ctor

        Args:
            **kwargs: Dict-like representation of the config file containing the distribution name and implementation (see EbsDistributions.py)
        """
        self.dstrb_v_ego = kwargs.get("dstrb_v_ego", self.dstrb_v_ego)
        self.dstrb_v_agent = kwargs.get("dstrb_v_agent", self.dstrb_v_agent)
        self.dstrb_clearance = kwargs.get("dstrb_clearance", self.dstrb_clearance)
        self.dstrb_a_brake = kwargs.get("dstrb_a_brake", self.dstrb_a_brake)
        # self.dstrb_mu_road		= kwargs.get("dstrb_mu_road", self.dstrb_mu_road)
        self.dstrb_ttc = kwargs.get("dstrb_ttc", self.dstrb_ttc)
        self.dstrb_fog_density = kwargs.get("dstrb_fog_density", self.dstrb_fog_density)
        self.dstrb_rain_intensity = kwargs.get(
            "dstrb_rain_intensity", self.dstrb_rain_intensity
        )
        self.dstrb_is_day = kwargs.get("dstrb_is_day", self.dstrb_is_day)

        self.min_mu_fric = kwargs.get("min_mu_fric", self.min_mu_fric)
        self.max_mu_fric = kwargs.get("max_mu_fric", self.max_mu_fric)

        self.t_simulation = kwargs.get("t_simulation", self.t_simulation)
        self.t_step = kwargs.get("t_step", self.t_step)
        self.poly_degree_first_detection = kwargs.get(
            "poly_degree_first_detection", self.poly_degree_first_detection
        )

        self.log_file_name = kwargs.get("log_file_name", self.log_file_name)
