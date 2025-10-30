# Script to cut the pre-processed (synced, cut) APDM and KiNetBlue files to n stride lengths and store them as new files
# Three datasets will be created:
# - APDM-APDM: APDM Signals and APDM Gait Events
# - KNB-KNB: KiNetBlue Signals and KiNetBlue Gait Events
# - KNB-APDM: KiNetBlue Signals and APDM Gait Events
from itertools import product

import numpy as np
import pandas as pd
from labtools.utils.hdf5 import load_dict_from_hdf5
from labtools.signal_processing.resampling import resize_signal

from utils import PATH_DATA_ROOT, SENSOR_LOCATIONS, STRIDE_COUNTS
import matplotlib

matplotlib.use("TkAgg")  # on older stacks "Qt5Agg" or "TkAgg" can be better

avg_samples_per_stride = 100


def get_gait_events(participant_id, session_id, path_gait_events) -> pd.DataFrame:
    path_file_gait_events = path_gait_events.joinpath(f"{participant_id}_{session_id}_gait_events.xlsx")
    return pd.read_excel(path_file_gait_events)


def make_signal(signal_data, sensor_location, t_start, t_end, new_length):
    t = signal_data[sensor_location]['timestamp'].astype(np.float64)
    t /= 1E3
    ii = np.where((t >= t_start) & (t <= t_end))
    gyr_cut = signal_data['left_foot']['gyr'][ii]
    gyr_cut_norm = resize_signal(gyr_cut, new_length)
    return gyr_cut_norm


def make_path(path_out_root, dataset_type, stride_count, participant_id, session_id, sensor_location):
    path_out = path_out_root.joinpath(dataset_type, str(stride_count).zfill(3), participant_id, session_id)
    path_out.mkdir(parents=True, exist_ok=True)
    filename = f"{sensor_location}.csv"
    path_file = path_out.joinpath(filename)
    return path_file


def make_signals(path_knb_file, path_apdm_file, participant_id, session_id, path_gait_events, path_out_root):
    gait_events = get_gait_events(participant_id, session_id, path_gait_events)
    data_knb = load_dict_from_hdf5(path_knb_file)
    data_apdm = load_dict_from_hdf5(path_apdm_file)
    for stride_count, sensor_location in product(STRIDE_COUNTS, SENSOR_LOCATIONS):
        t_knb_start = gait_events[f'ic_times_knb_ms'][0]
        t_knb_end = gait_events[f'ic_times_knb_ms'][stride_count]
        t_apdm_start = gait_events[f'ic_times_apdm_ms'][0]
        t_apdm_end = gait_events[f'ic_times_apdm_ms'][stride_count]
        # Get the indices for KiNetBlue data
        normalized_length = avg_samples_per_stride * stride_count
        gyr_knb_knb = make_signal(data_knb, sensor_location, t_knb_start, t_knb_end, normalized_length)
        gyr_apdm_apdm = make_signal(data_apdm, sensor_location, t_apdm_start, t_apdm_end, normalized_length)
        gyr_knb_apdm = make_signal(data_knb, sensor_location, t_apdm_start, t_apdm_end, normalized_length)

        # Make output directories and save files
        path_out_knb_knb = make_path(path_out_root, "KNB_KNB", stride_count, participant_id, session_id, sensor_location)
        path_out_apdm_apdm = make_path(path_out_root, "APDM_APDM", stride_count, participant_id, session_id, sensor_location)
        path_out_knb_apdm = make_path(path_out_root, "KNB_APDM", stride_count, participant_id, session_id, sensor_location)
        fmt = '%1.12f'
        np.savetxt(path_out_knb_knb, gyr_knb_knb, delimiter=",", fmt=fmt, header="x, y, z", comments="")  # 12 decimal places should be enough
        np.savetxt(path_out_apdm_apdm, gyr_apdm_apdm, delimiter=",", fmt=fmt, header="x, y, z", comments="")
        np.savetxt(path_out_knb_apdm, gyr_knb_apdm, delimiter=",", fmt=fmt, header="x, y, z", comments="")


def main():
    path_data_root = PATH_DATA_ROOT.joinpath('sync_cut')
    path_gait_events = PATH_DATA_ROOT.joinpath('gait_events')
    path_out_root = PATH_DATA_ROOT.joinpath('datasets')
    for path_participant in path_data_root.iterdir():
        p_id = path_participant.stem
        for path_session in path_participant.iterdir():
            s_id = path_session.stem
            path_knb_file = list(path_session.glob("*KiNetBlue.hdf5"))[0]
            path_apdm_file = list(path_session.glob("*APDM.hdf5"))[0]
            print(f"Making signals for {p_id} - {s_id} ...")
            make_signals(path_knb_file, path_apdm_file, p_id, s_id, path_gait_events, path_out_root)


if __name__ == '__main__':
    main()
