from itertools import product

import numpy as np
import pandas as pd

from event_detection import get_running_events
from utils import *

import matplotlib.pyplot as plt


def get_data_sync_cut(participant_id: str, session_id: str, system: str) -> dict | None:
    path_sync_cut_root = PATH_DATA_ROOT.joinpath("sync_cut")
    path_file_parent = path_sync_cut_root.joinpath(participant_id, session_id)
    filename = f"{participant_id}_{session_id}_RunTM_{system}.hdf5"
    path_file = path_file_parent.joinpath(filename)
    if not path_file.exists():
        return None
    data = load_dict_from_hdf5(path_file)
    return data


def make_foot_events(data: dict):
    left_foot = data.get('left_foot')
    t = left_foot['timestamp']
    t = t / 1E3  # convert to milliseconds
    gait_events = get_running_events(t=t,
                                     acc=left_foot['acc'],
                                     gyr=left_foot['gyr'],
                                     sensor_location='foot',
                                     side='left')
    ics = gait_events['ic']
    t_ics = t[ics]
    return t_ics


def best_slice(A, B):
    n = len(A)
    diffs = [np.sum((A - B[i:i + n]) ** 2) for i in range(len(B) - n + 1)]
    i_min = np.argmin(diffs)
    return i_min, B[i_min:i_min + n]


def main():
    path_data_root = PATH_DATA_ROOT.joinpath('sync_cut')
    path_events_out = PATH_DATA_ROOT.joinpath('gait_events')
    min_ic_count = 100000
    worst_average_difference = 0
    for p_id, s_id in product(SUBJECTS, SESSIONS):
        print(f'Processing {p_id} - {s_id} ...')
        data_apdm = get_data_sync_cut(p_id, s_id, system="APDM")
        data_knb = get_data_sync_cut(p_id, s_id, system="KiNetBlue")
        if data_apdm is None:
            continue
        if data_knb is None:
            continue

        t_ics_apdm = make_foot_events(data_apdm)
        t_ics_knb = make_foot_events(data_knb)

        # remove the fist and last ten of the apdm ics to avoid edge effects
        t_ics_apdm = t_ics_apdm[10:-10]
        # Make sure knb has at least as many ics as apdm
        ics_count = len(t_ics_apdm)
        if ics_count >= len(t_ics_knb):
            raise ValueError(f'Not enough ICs in KiNetBlue data for {p_id} - {s_id}.')

        # find the len(t_ics_apdm) closest ics in t_ics_knb
        i_start, t_ics_knb_best = best_slice(t_ics_apdm, t_ics_knb)
        t_ics_knb = t_ics_knb_best

        df = pd.DataFrame({
            'ic_times_apdm_ms': t_ics_apdm,
            'ic_times_knb_ms': t_ics_knb
        })
        # add "difference_ms" column
        df['difference_ms'] = df['ic_times_knb_ms'] - df['ic_times_apdm_ms']
        max_diff = df['difference_ms'].abs().max()
        if max_diff > 10:
            warnings.warn(f'Large maximum difference of {max_diff} ms for {p_id} - {s_id}.')
        avg_diff = df['difference_ms'].mean()
        if abs(avg_diff) > abs(worst_average_difference):
            worst_average_difference = avg_diff
            print(f'New worst average difference: {worst_average_difference} ms for {p_id} - {s_id}')
        make_plot = False
        if make_plot:
            gyr_apdm = data_apdm['left_foot']['gyr'][:, 0] * (180.0 / 3.141592653589793)
            gyr_knb = data_knb['left_foot']['gyr'][:, 0]
            t_apdm = data_apdm['left_foot']['timestamp'] / 1E3
            t_knb = data_knb['left_foot']['timestamp'] / 1E3
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_apdm, gyr_apdm, label='APDM Gyr X', color='blue')
            ax.plot(t_knb, gyr_knb, label='KiNetBlue Gyr X', color='orange', linestyle='--')
            for t_ic in t_ics_apdm:
                ax.axvline(t_ic, color='green', linestyle=':', alpha=0.5)
            for t_ic in t_ics_knb:
                ax.axvline(t_ic, color='red', linestyle=':', alpha=0.5)
            plt.show()

        if len(df) < min_ic_count:
            min_ic_count = len(df)
        filename = f'{p_id}_{s_id}_gait_events.hdf5'
        path_file_out = path_events_out.joinpath(filename)
        path_events_out.mkdir(parents=True, exist_ok=True)
        # safe to excel
        df.to_excel(path_file_out.as_posix().replace('.hdf5', '.xlsx'), index=False)
    print(f'Minimum number of ICs across all sessions: {min_ic_count}')


if __name__ == '__main__':
    main()
