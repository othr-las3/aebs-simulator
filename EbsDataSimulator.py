"""Provides the actual implementation of the data generator/simulator.
    Expectation is a car-follows-car scenario.
"""
import copy
import numpy as np

import EbsMetrics
from EbsLogger import EbsLogger
from SimulatorConfig import SimulatorConfig
from ParameterContainer import InitParameter, StepParameter


class EbsDataSimulator:

    """Provides the actual implementation of the data generator/simulator.
        Expectation is a car-follows-car scenario.
        The ego vehicle is behind the agent vehicle.
        At the start of a simulation run each relevant paramter (e.g., velocities, clearance...) get sampled from a configured distribution.
        The simulation is executed either until a collison occures (clearance <= 0m) or until the simulation time elapsed.
        If breaking is initiated it is expected to be constant with a calculated decceleration based on wheather conditons and a max. break decc..
        Breaking is constant, linear, and instantaneous.
        Log data is collected and returned.

    Attributes:
        config (SimulatorConfig): Parameterization of the car-follows-car scenario.
        init_params (InitParameter): Sampled parameter values at simulation start.
        is_collision (bool): Indicator if a collision occured.
        is_resimulation (bool): Whether or not the current configuration has fixed parameters (no sampling for those).
        log_data (dict): Logged data throughout simulation run
        log_entries (list<str>): Which data should be logged.
        log_is_active (bool): Whether or not rutime data should be logged
        log_resolution (int): Rounding of numeric data.
        logger (EbsLogger): Handle for Logger.
    """

    config = None  # SimulatorConfig instance
    is_collision = None  # bool
    is_resimulation = False

    log_resolution = 2  # int, decimals logged data will be rounded to
    log_is_active = None  # bool
    log_entries = [
        "t_elapsed",
        "cur_v_ego",
        "cur_v_agent",
        "cur_x_clearance",
        "sampled_a_brake",
        "cur_a_brake",
        "is_braking",
        "cur_collision_energy",
        "is_collision",
    ]
    log_data = None  # dicitonary containing data keyed by entry name
    logger = None  # EbsLogger instance

    init_params = None  # start params sampled for the current run

    def __init__(self, simulator_config=None, log_is_active=False):
        """Ctor

        Args:
            simulator_config (SimulatorConfig, optional): Parameterization of the car-follows-car scenario.
            log_is_active (bool, optional): Whether or not rutime data should be logged

        Raises:
            ValueError: Indicator if a valid configuration is available.
        """
        if simulator_config and not isinstance(simulator_config, SimulatorConfig):
            raise ValueError(f"To run this simualtor a simulation config is required.")

        self.configure_simulation(simulator_config)
        self.log_is_active = log_is_active

    def run_simulation(self, config=None, run_id=None):
        """Simulator 'main' function. Manages the actual simulation by governing all simulation steps in a structured order.

        Args:
            config (SimulatorConfig, optional): Parameterization of the car-follows-car scenario.
            run_id (int, optional): Used if mulithreaded simulation is enabled (concurrent and independent simulation runs)

        Returns:
            dict: Aggregated simulation data.
        """
        run_id = run_id if run_id else 0

        self.setup_logging()
        self.configure_simulation(config)
        self.init_params = self.sample_run_params(run_id)

        step_params = StepParameter(
            cur_v_ego=self.init_params.sampled_v_ego,
            cur_v_agent=self.init_params.sampled_v_agent,
            cur_x_clearance=self.init_params.sampled_clearance,
            is_braking=False,
            cur_collision_energy=0.0,
        )

        final_step_params = step_params

        # run simulation
        for t_elapsed in np.arange(
            start=0.0,
            stop=self.config.t_simulation + self.config.t_step,
            step=self.config.t_step,
        ):
            step_params = self.step(step_params, self.init_params, self.config)

            if self.log_is_active:
                self.log_step_data(t_elapsed, step_params, self.init_params)

            final_step_params = step_params
            stop_simulation = self.monitor(
                step_params
            )  # check if collision OR ego stopped
            if stop_simulation:
                if self.is_collision:
                    final_step_params.cur_collision_energy = (
                        EbsMetrics.calc_collision_energy(
                            final_step_params.cur_v_ego, final_step_params.cur_v_agent
                        )
                    )
                break

        if self.log_is_active:
            self.write_log_data()

        return self.build_simulation_result(final_step_params)

    def run_resimulation(self, resim_config, config=None, run_id=None):
        """Simulator 'secondary' function. Manages the instantiation of a simulation if some parameters are fixed.

        Args:
            (ReSimulationConfig): Parameterization of a partially fixed-value car-follows-car scenario.
            config (SimulatorConfig, optional): Parameterization of the car-follows-car scenario.
            run_id (int, optional): Used if mulithreaded simulation is enabled (concurrent and independent simulation runs)

        Returns:
            dict: Aggregated simulation data via 'run_simulation'
        """
        self.is_resimulation = True
        if config is None:
            config = SimulatorConfig()  # use default

        resim_params = resim_config.fixed_params

        # fix distributions
        config.dstrb_a_brake = (
            config.dstrb_a_brake
            if "dstrb_a_brake" not in resim_params
            or resim_params["dstrb_a_brake"] is None
            else resim_params["dstrb_a_brake"]
        )
        config.dstrb_v_ego = (
            config.dstrb_v_ego
            if "dstrb_v_ego" not in resim_params or resim_params["dstrb_v_ego"] is None
            else resim_params["dstrb_v_ego"]
        )
        config.dstrb_v_agent = (
            config.dstrb_v_agent
            if "dstrb_v_agent" not in resim_params
            or resim_params["dstrb_v_agent"] is None
            else resim_params["dstrb_v_agent"]
        )
        config.dstrb_ttc = (
            config.dstrb_ttc
            if "dstrb_ttc" not in resim_params or resim_params["dstrb_ttc"] is None
            else resim_params["dstrb_ttc"]
        )

        config.dstrb_fog_density = (
            config.dstrb_fog_density
            if "dstrb_fog_density" not in resim_params
            or resim_params["dstrb_fog_density"] is None
            else resim_params["dstrb_fog_density"]
        )
        config.dstrb_rain_intensity = (
            config.dstrb_rain_intensity
            if "dstrb_rain_intensity" not in resim_params
            or resim_params["dstrb_rain_intensity"] is None
            else resim_params["dstrb_rain_intensity"]
        )
        config.dstrb_is_day = (
            config.dstrb_is_day
            if "dstrb_is_day" not in resim_params
            or resim_params["dstrb_is_day"] is None
            else resim_params["dstrb_is_day"]
        )

        config.dstrb_clearance = (
            config.dstrb_clearance
            if "dstrb_clearance" not in resim_params
            or resim_params["dstrb_clearance"] is None
            else resim_params["dstrb_clearance"]
        )

        # provide updated configuration to the regular simulation run.
        return self.run_simulation(config=config, run_id=run_id)

    def sample_run_params(self, run_id=None):
        """Samples the initial values to instantiate a parameterized simulation run.

        Args:
            run_id (int, optional): Used if mulithreaded simulation is enabled (concurrent and independent simulation runs).

        Returns:
            InitParameter: Start parameters for a simulation.
        """
        sampled_a_brake = -1.0 * self.config.dstrb_a_brake.sample()
        sampled_v_ego = self.config.dstrb_v_ego.sample() / 3.6  # cur specified in km/h
        sampled_v_agent = (
            self.config.dstrb_v_agent.sample() / 3.6
        )  # cur specified in km/h
        # if v_agent_cur >= v_ego_cur: check only needed if bad combination.. and if we want a crash to be possible in the first place

        # Clearance needs to be greater than d_brake and or greater TTC (otherwise it is already a crash/cut-in)
        # sampled_clearance = self.config.dstrb_clearance.sample()
        sampled_ttc_target = self.config.dstrb_ttc.sample()

        sampled_fog = self.config.dstrb_fog_density.sample()
        sampled_rain = self.config.dstrb_rain_intensity.sample()
        is_day = self.config.dstrb_is_day.sample()

        real_friction = EbsMetrics.calc_friction_on_rain(
            sampled_rain, self.config.min_mu_fric, self.config.max_mu_fric
        )

        real_brake = EbsMetrics.calc_a_brake_real(
            sampled_a_brake, real_friction, self.config.max_mu_fric
        )
        dist_first_detection = EbsMetrics.calc_x_first_detection(
            sampled_fog,
            sampled_rain,
            is_day,
            poly_degree=self.config.poly_degree_first_detection,
        )
        dist_x_activation = EbsMetrics.calc_x_activation(
            sampled_v_ego, sampled_v_agent, sampled_ttc_target
        )
        dist_x_braking = EbsMetrics.calc_x_braking(
            dist_first_detection, dist_x_activation
        )

        dist_buffer_spawning = 10
        sampled_clearance = (
            self.config.dstrb_clearance.sample()
            if self.is_resimulation
            else dist_x_braking + dist_buffer_spawning
        )  ##we dont need to sample since everything before x_braking is irrelevant for our usecase

        return InitParameter(
            run_id=run_id,
            sampled_v_ego=sampled_v_ego,
            sampled_v_agent=sampled_v_agent,
            sampled_clearance=sampled_clearance,
            sampled_a_brake=sampled_a_brake,
            sampled_ttc_target=sampled_ttc_target,
            sampled_fog=sampled_fog,
            sampled_rain=sampled_rain,
            is_day=is_day,
            real_friction=real_friction,
            real_brake=real_brake,
            dist_first_detection=dist_first_detection,
            dist_x_activation=dist_x_activation,
            dist_x_braking=dist_x_braking,
        )

    def step(self, step_params, init_params, config):
        """Executes all calculations of a single step in a structured way (evaluating the causal mechanisms in correct order).

        Args:
            step_params (StepParameter): Simulation parameters of the last step which are needed to run all calculations for this step.
            init_params (InitParameter): Sampled parameter values at simulation start.
            config (SimulatorConfig): Parameterization of the car-follows-car scenario.

        Returns:
            StepParameter: Current parameterization of the simulator for this step.
        """
        is_braking = EbsMetrics.is_ebs_active(
            step_params.cur_x_clearance,
            init_params.dist_x_braking,
            init_params.dist_first_detection,
        )

        cur_a_brake = init_params.real_brake if is_braking else 0.0
        cur_v_ego = EbsMetrics.calc_v_after_accelaration(
            step_params.cur_v_ego, a=cur_a_brake, t_delta=config.t_step
        )
        cur_v_agent = EbsMetrics.calc_v_after_accelaration(
            step_params.cur_v_agent, a=0.0, t_delta=config.t_step
        )
        s_ego = EbsMetrics.calc_dist_moved(
            step_params.cur_v_ego, a=cur_a_brake, t_delta=config.t_step
        )
        s_agent = EbsMetrics.calc_dist_moved(
            step_params.cur_v_agent, a=0, t_delta=config.t_step
        )
        s_moved = np.abs(s_ego - s_agent)
        s_moved = s_moved if s_agent >= s_ego else -1.0 * s_moved
        cur_x_clearance = step_params.cur_x_clearance + s_moved

        return StepParameter(
            cur_v_ego=cur_v_ego,
            cur_v_agent=cur_v_agent,
            cur_x_clearance=cur_x_clearance,
            is_braking=is_braking,
            cur_a_brake=cur_a_brake,
            cur_collision_energy=0.0,
        )  # collision energy only if accident

    def monitor(self, step_params):
        """Evaluates special if special conditions occured (e.g., collision) and manages logging of data.

        Args:
            step_params (StepParameter): Current parameterization of the simulator for this step.

        Returns:
            bool: Indicator if the simulation should be continued or not.
        """
        self.is_collision = self.check_collision(step_params.cur_x_clearance, buffer=0)

        if self.log_is_active:
            self.update_log_data("is_collision", self.is_collision)

        stop_simulation = (
            True if self.is_collision else self.check_sim_stop_reasons(step_params)
        )

        return stop_simulation

    def configure_simulation(self, config=None):
        """Summary

        Args:
            config (SimulatorConfig, optional): Parameterization of the car-follows-car scenario. If not provided a default param. is used.

        Raises:
            ValueError: Raised if an invalid configuration is provided
        """
        if config:
            if not isinstance(config, SimulatorConfig):
                raise ValueError(f"Invalid config passed for current simulation run.")
            self.config = config

        else:
            self.config = SimulatorConfig()  # use default

        # configure helper variables
        self.is_collision = False

    def check_collision(self, x_clearance, buffer=0):
        """Check for a collision.

        Args:
            x_clearance (float): Current clearance (bumper to bumper)
            buffer (int, optional): Optional parameter if a collision should be considered if a min. clearance is violated.

        Returns:
            bool: Indicator if the simulation should be continued or not.
        """
        return True if x_clearance <= buffer else False

    def check_sim_stop_reasons(self, step_params):
        """Manger function that evaluates all conditions why a simulation run should be over.

        Args:
            step_params (StepParameter): Current parameterization of the simulator for this step.

        Returns:
            bool: Indicator if the simulation should be continued or not.
        """
        stop_simulation = False

        if step_params.cur_v_ego:
            stop_simulation = True if step_params.cur_v_ego <= 0.0 else False

        return stop_simulation

    def setup_logging(self):
        """Initializes a logger for an individual simulation run."""
        self.log_data = {}
        for entry in self.log_entries:
            self.log_data.setdefault(entry, [])

        if self.log_is_active:
            self.logger = EbsLogger(
                data_header=self.log_entries,
                logger_name=self.config.log_file_name,
                file_name=self.config.log_file_name,
            )

    def update_log_data(self, name, value):
        """Updates (i.e., adds one row of data for the current step) the data which will be logged at the end of the simulation run.

        Args:
            name (str): Name of the variable to be logged.
            value (*): Value to be logged

        Raises:
            ValueError: Raised if a variable should be logged that was not configured as input to the logger.
        """
        if name not in self.log_data.keys():
            raise ValueError(
                f"To log data for {str(name)}, an appropriate configuration of the logging header is required."
            )

        if isinstance(value, (list, tuple, np.ndarray)):
            self.log_data[name].extend(value)
        else:
            self.log_data[name].extend([value])

    def log_step_data(self, t_elapsed, step_params, init_params):
        """Transforms current step data into the used log format.

        Args:
            t_elapsed (double): Elapsed time until this step.
            step_params (StepParameter): Simulation parameters of the last step which are needed to run all calculations for this step.
            init_params (InitParameter): Sampled parameter values at simulation start.
        """
        self.update_log_data("t_elapsed", t_elapsed)
        self.update_log_data("cur_v_ego", step_params.cur_v_ego)
        self.update_log_data("cur_v_agent", step_params.cur_v_agent)
        self.update_log_data("cur_x_clearance", step_params.cur_x_clearance)
        self.update_log_data("sampled_a_brake", init_params.sampled_a_brake)
        self.update_log_data("cur_a_brake", step_params.cur_a_brake)
        coll_ener = (
            0
            if step_params.cur_x_clearance >= 0
            else EbsMetrics.calc_collision_energy(
                step_params.cur_v_ego, step_params.cur_v_agent
            )
        )
        self.update_log_data("cur_collision_energy", coll_ener)
        self.update_log_data("is_braking", step_params.is_braking)

    def build_simulation_result(self, step_params):
        """Aggregate and transform logged data into the final output format

        Args:
            step_params (StepParameter): Simulation parameters of the last step which are needed to run all calculations for this step.

        Returns:
            dict: Parameters describing the executed simulation run.
        """
        results = {
            "run_id": self.init_params.run_id
            if isinstance(self.init_params.run_id, (list, tuple, np.ndarray))
            else [self.init_params.run_id],
            "t_simulation": self.config.t_simulation
            if isinstance(self.config.t_simulation, (list, tuple, np.ndarray))
            else [self.config.t_simulation],
            "t_step": self.config.t_step
            if isinstance(self.config.t_step, (list, tuple, np.ndarray))
            else [self.config.t_step],
            "poly_degree_first_detection": self.config.poly_degree_first_detection
            if isinstance(
                self.config.poly_degree_first_detection, (list, tuple, np.ndarray)
            )
            else [self.config.poly_degree_first_detection],
            "sampled_v_ego": self.init_params.sampled_v_ego
            if isinstance(self.init_params.sampled_v_ego, (list, tuple, np.ndarray))
            else [self.init_params.sampled_v_ego],
            "sampled_v_agent": self.init_params.sampled_v_agent
            if isinstance(self.init_params.sampled_v_agent, (list, tuple, np.ndarray))
            else [self.init_params.sampled_v_agent],
            "sampled_clearance": self.init_params.sampled_clearance
            if isinstance(self.init_params.sampled_clearance, (list, tuple, np.ndarray))
            else [self.init_params.sampled_clearance],
            "sampled_ttc_target": self.init_params.sampled_ttc_target
            if isinstance(
                self.init_params.sampled_ttc_target, (list, tuple, np.ndarray)
            )
            else [self.init_params.sampled_ttc_target],
            "sampled_a_brake": self.init_params.sampled_a_brake
            if isinstance(self.init_params.sampled_a_brake, (list, tuple, np.ndarray))
            else [self.init_params.sampled_a_brake],
            "sampled_fog": self.init_params.sampled_fog
            if isinstance(self.init_params.sampled_fog, (list, tuple, np.ndarray))
            else [self.init_params.sampled_fog],
            "sampled_rain": self.init_params.sampled_rain
            if isinstance(self.init_params.sampled_rain, (list, tuple, np.ndarray))
            else [self.init_params.sampled_rain],
            "is_day": self.init_params.is_day
            if isinstance(self.init_params.is_day, (list, tuple, np.ndarray))
            else [self.init_params.is_day],
            "real_friction": self.init_params.real_friction
            if isinstance(self.init_params.real_friction, (list, tuple, np.ndarray))
            else [self.init_params.real_friction],
            "real_brake": self.init_params.real_brake
            if isinstance(self.init_params.real_brake, (list, tuple, np.ndarray))
            else [self.init_params.real_brake],
            "dist_first_detection": self.init_params.dist_first_detection
            if isinstance(
                self.init_params.dist_first_detection, (list, tuple, np.ndarray)
            )
            else [self.init_params.dist_first_detection],
            "dist_x_activation": self.init_params.dist_x_activation
            if isinstance(self.init_params.dist_x_activation, (list, tuple, np.ndarray))
            else [self.init_params.dist_x_activation],
            "dist_x_braking": self.init_params.dist_x_braking
            if isinstance(self.init_params.dist_x_braking, (list, tuple, np.ndarray))
            else [self.init_params.dist_x_braking],
            "collision_energy": step_params.cur_collision_energy
            if isinstance(step_params.cur_collision_energy, (list, tuple, np.ndarray))
            else [step_params.cur_collision_energy],
            "is_collision": int(self.is_collision)
            if isinstance(self.is_collision, (list, tuple, np.ndarray))
            else [int(self.is_collision)],
        }

        return results

    def write_log_data(self):
        """Write output data to the configured log file."""
        # additional preprocessing could be done here
        # additional calculation e.g. of energy could be done her
        # this would render it a numpy array operation
        # instead of a single calc at each time step -> performance
        self.logger.write(self.log_data, decimals=self.log_resolution)
