"""Re-simulate available simulation runs based on an aggregated logfile.

Attributes:
    var_mapping (TYPE): Description
    var_preproc_factor (TYPE): Description
"""
# Provide a aggregated results file and re-simulate each run with
# a set of fixed values

import logging
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from EbsLogger import EbsLogger
from SimulatorConfig import SimulatorConfig
from EbsDataSimulator import EbsDataSimulator
from ReSimulationConfig import ReSimulationConfig


# from name in file to name in config
var_mapping = {
    "sampled_v_ego": "dstrb_v_ego",
    "sampled_v_agent": "dstrb_v_agent",
    "sampled_clearance": "dstrb_clearance",
    "sampled_a_brake": "dstrb_a_brake",
    "sampled_ttc_target": "dstrb_ttc",
    "sampled_fog": "dstrb_fog_density",
    "sampled_rain": "dstrb_rain_intensity",
    "is_day": "dstrb_is_day",
}

var_preproc_factor = {
    "sampled_v_ego": 3.6,
    "sampled_v_agent": 3.6,
    "sampled_clearance": 1.0,
    "sampled_a_brake": -1.0,
    "sampled_ttc_target": 1.0,
    "sampled_fog": 1.0,
    "sampled_rain": 1.0,
    "is_day": 1.0,
}


def provide_iterable(df, resim_config):
    """Summary

    Args:
        df (TYPE): Description
        resim_config (TYPE): Description

    Yields:
        TYPE: Description
    """
    for series in df.iterrows():
        yield resim_config, series


def async_process(params):
    """Summary

    Args:
        params (TYPE): Description

    Returns:
        TYPE: Description
    """
    resim_config, series = params
    idx, data = series

    # fix all params based on the original run
    custom_resim_config = ReSimulationConfig()
    for log_var, dist_var in var_mapping.items():
        resim_val = data[log_var] * var_preproc_factor[log_var]
        custom_resim_config.add_fixed_vals(**{dist_var: resim_val})

    # override with inteded fixed params for resimulation
    custom_resim_config.fixed_params.update(resim_config.fixed_params)
    sim_config = SimulatorConfig(**{"log_file_name": f"Param_log_resim_{idx}.csv"})
    simulator = EbsDataSimulator(sim_config, log_is_active=False)

    return simulator.run_resimulation(
        resim_config=custom_resim_config, config=sim_config, run_id=idx
    )


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count() if multiprocessing.cpu_count() else 2
    baseline_simulation_path = r"..\AggregatedResults.csv"
    result_file_name = "ResimulatedResults.csv"

    df = pd.read_csv(baseline_simulation_path)
    pool = Pool(num_cores)

    resim_config = ReSimulationConfig(
        **{"dstrb_v_ego": 100.0, "dstrb_fog_density": 85.0}
    )

    results = pool.map(
        async_process, iterable=provide_iterable(df, resim_config), chunksize=num_cores
    )

    print(f"A total of {len(results)} results were returned")

    aggregated_data = {}
    [aggregated_data.setdefault(k, []) for k in results[0].keys()]

    for sim_result in results:
        for k, v in sim_result.items():
            aggregated_data[k].extend(list(v))

    result_logger = EbsLogger(
        logger_name="ResimulatedResults",
        file_name=result_file_name,
        data_header=list(aggregated_data.keys()),
        mirror=False,
    )
    result_logger.write(aggregated_data, decimals=2)
    logging.shutdown()
