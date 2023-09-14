"""Central file for the Monte-Carlo based simulator.
    Simulation here is to be understood as a continous evaluation of the implemented causal mechanisms.
    You can provide the number of samples (i.e., brute-force, iterative, step-by-step evaluation of a car follows car scenarios),
    their respective output directory, and whether or not you want to use multi-threading for the calculation.
    Additionally a re-run option is available. 
    Re-runs can be parameterized and can be thought of an intervention (i.e., some parameters are fixed, some are still sampled).
"""
import os
import logging
import shutil
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing


from EbsLogger import EbsLogger
from SimulatorConfig import SimulatorConfig
from EbsDataSimulator import EbsDataSimulator
from ReSimulationConfig import ReSimulationConfig


def simulate_parallel(
    total_iterations, config=None, log_data=True, show_progress=True, resim_config=None
):
    """Coordinator method for the (parallel) execution of the request number of simulation runs.

    Args:
        total_iterations (int): Number of simulation runs (nr. of Monte-Carlo sampling iterations).
        config (SimulatorConfig, optional): Parameterization of the car-follows-car scenario.
        log_data (bool, optional): Whether or not to generate individual log files per run.
        show_progress (bool, optional): Whether or not to show a progress bar (cur. iter. / total iter.)
        resim_config (ReSimulationConfig, optional): Parameterization of a partially fixed-value car-follows-car scenario.

    Returns:
        dict: Dictionary containing collected simulation data.
    """

    def run(iteration, config, resim_config=None):
        """Hypervisor managing the dispatchment and aggregation of individual simulator runs.

        Args:
            iteration (int): Current iteration (numeric ID from 1 to max. iterations requested)
            config (SimulatorConfig): Parameterization of the car-follows-car scenario.
            resim_config (ReSimulationConfig, optional): Parameterization of a partially fixed-value car-follows-car scenario.

        Returns:
            dict: Dictionary containing collected simulation data.
        """
        sim_config = (
            config
            if config
            else SimulatorConfig(**{"log_file_name": f"Param_log_{iteration}.csv"})
        )
        simulator = EbsDataSimulator(sim_config, log_is_active=log_data)

        if resim_config is None:
            return simulator.run_simulation(run_id=iteration)

        return simulator.run_resimulation(
            resim_config=resim_config, config=sim_config, run_id=iteration
        )

    num_cores = multiprocessing.cpu_count() if multiprocessing.cpu_count() else 2
    results = None
    if show_progress:
        results = Parallel(n_jobs=num_cores)(
            delayed(run)(i, config, resim_config) for i in trange(total_iterations)
        )
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(run)(i, config, resim_config) for i in range(total_iterations)
        )

    aggregated_data = {}
    [aggregated_data.setdefault(k, []) for k in results[0].keys()]

    for sim_result in results:
        for k, v in sim_result.items():
            aggregated_data[k].extend(list(v))

    return aggregated_data


if __name__ == "__main__":
    # clear "old" simulation results
    if os.path.isdir("./Logs"):
        shutil.rmtree("./Logs")

    result_file_name = "AggregatedResults.csv"

    # ----------- Example data generation --------------
    print("-" * 50)
    print("Generate data")
    aggregated_results = simulate_parallel(
        total_iterations=int(10),
        config=None,
        log_data=True,
        show_progress=True,
        # resim_config= ReSimulationConfig(**{"dstrb_v_ego": 100.0, "dstrb_fog_density":85.0})
    )
    result_logger = EbsLogger(
        logger_name="AggregatedResults",
        file_name=result_file_name,
        data_header=list(aggregated_results.keys()),
        mirror=False,
    )
    result_logger.write(aggregated_results, decimals=2)

    logging.shutdown()
