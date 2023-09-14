"""Provides the parameterization for a re-simulation.
    That is: all causal mechanisms stay intact, non-provided parameters are still sampled.
    If parameters are provided (i.e., fixed at a specific value), then they replace the values 
    provided via initial sampling of params. at simulation start (which in turn is parameterized via SimulatorConfigs).
"""


from EbsDistributions import *
from SimulatorConfig import SimulatorConfig


class ReSimulationConfig:

    """Class that manages a set of fixable parameters.
        These parameters are used as static values for an initialization of simulation runs.
        If provided they replace the respective init values that are regularly provided via sampling.

    Attributes:
        fixable_params (list<str>): List of parameter distributions which can be fixed to a specific value. These parameters are defined in EbsDistributions.py.
        fixed_params (dict): Dictionary containing the actual fixed params (and their respective value) for a simluation.
    """

    fixable_params = [
        "dstrb_v_ego",
        "dstrb_v_agent",
        "dstrb_clearance",
        "dstrb_a_brake",
        "dstrb_ttc",
        "dstrb_fog_density",
        "dstrb_rain_intensity",
        "dstrb_is_day",
    ]

    fixed_params = None

    def __init__(self, **kwargs):
        """Ctor for this container class.

        Args:
            **kwargs: Description
        """
        self.fixed_params = {}
        self.add_fixed_vals(**kwargs)

    def add_fixed_vals(self, **kwargs):
        """Ínitializer method for the container class.
            Assures that to-be-fixed values are indeed eligible (i.e., represent a valid parameter distribution as defined in EbsDistributions.py)

        Args:
            **kwargs: Dict-like configuration of to-be-fixed parameters and their target value.

        Raises:
            ValueError: Raised if to-be-fixed parameter is not eligible (i.e., not linked to a valid distribution)
        """
        for param in kwargs:
            if param not in self.fixable_params:
                raise ValueError(
                    f"Requested parameter: {param} can not be fixed for re-simulation."
                )

            fixed_value = kwargs.get(param, None)
            self.fixed_params[param] = (
                Singleton(fixed_value=fixed_value) if fixed_value is not None else None
            )
