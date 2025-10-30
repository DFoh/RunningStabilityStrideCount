import os
from itertools import product
from pathlib import Path

from labtools.analyses.LocalDynamicStability.analysis import LDSAnalysis
from labtools.analyses.LocalDynamicStability.plotting import plot_divergence_curves

from logger import logger

from utils import PATH_DATA_ROOT, SENSOR_LOCATIONS, STRIDE_COUNTS

if __name__ == '__main__':
    logger.info("Setting up data directories...")
    path_datasets = PATH_DATA_ROOT.joinpath("datasets")
    if not path_datasets.exists():
        raise FileNotFoundError(f"Datasets path not found: {path_datasets}")
    path_results = PATH_DATA_ROOT.joinpath("results")
    path_results.mkdir(parents=True, exist_ok=True)

    dataset_names = ["APDM_APDM", "KNB_KNB", "KNB_APDM"]

    for dataset_name, stride_count in product(dataset_names, STRIDE_COUNTS):
        # stride_count_str = str(stride_count).zfill(3)
        stride_count_str = "150"  # todo: REMOVE AFTER TESTING!!!
        path_data_in = path_datasets.joinpath(dataset_name, stride_count_str)
        logger.debug("Input data path: %s", path_data_in)
        path_data_out = path_results.joinpath(dataset_name, stride_count_str)
        path_data_out.mkdir(parents=True, exist_ok=True)
        logger.debug("Output data path: %s", path_data_out)

        for sensor_location in SENSOR_LOCATIONS:
            analysis = LDSAnalysis(path_data_in=path_data_in,
                                   path_data_out=path_data_out,
                                   sensor_location=sensor_location,
                                   force_recalculate=True)
            logger.info(f"Dataset: {dataset_name}, Strides: {stride_count}, Sensor: {sensor_location}")
            analysis.compute_time_delays()
            analysis.time_delay_summary()
            analysis.compute_embedding_dimensions()
            analysis.embedding_dimension_summary()
            analysis.compute_divergence_curves()
            divergence_curves = analysis.divergence_curves
            plot_divergence_curves(divergence_curves=analysis.divergence_curves,
                                   path_data_out=analysis.path_data_out,
                                   location_string=analysis.sensor_location)

            # analysis.set_fit_interval(end=30)  # start=0 per default
            # analysis.compute_divergence_exponents()
            # print(analysis.divergence_exponents.head())

            break

        break
