# cut the previously synced apdm and kinetblue files right before the treadmill run starts
from itertools import product

import pandas as pd
from labtools.utils.cutting_tool import InteractiveCuttingTool
import matplotlib.pyplot as plt
import numpy as np
from labtools.utils.hdf5 import save_dict_to_hdf5

from pre_processing.cut_apdm_files import get_cut_marks
from utils import *


def load_cut_marks_final() -> pd.DataFrame | None:
    path_file = get_cut_marks_final_path()
    if path_file.exists():
        return pd.read_excel(path_file)


def get_cut_marks_final_path() -> Path:
    return PATH_DATA_ROOT.joinpath('final_cut_timestamps.xlsx')


def save_cut_marks_final(cut_marks: pd.DataFrame):
    path_file = get_cut_marks_final_path()
    # sort df by participant_id then session_id
    cut_marks.sort_values(by=["participant_id", "session_id"], inplace=True)
    cut_marks.to_excel(path_file, index=False)


def get_path_knb_sync_root():
    return PATH_DATA_ROOT.joinpath("sync", "KiNetBlue")


def get_knb_data(participant_id: str, session_id: str) -> dict | None:
    path_sync_root = get_path_knb_sync_root()
    path_file_parent = path_sync_root.joinpath(participant_id, session_id)
    filename = f"{participant_id}_{session_id}_RunTM_sync.hdf5"
    path_file = path_file_parent.joinpath(filename)
    if not path_file.exists():
        return
    data = load_dict_from_hdf5(path_file)
    return data


def get_timestamps(df_existing_timestamps: pd.DataFrame = None):
    cols = ["participant_id", "session_id", "timestamp_start", "timestamp_end"]
    df = pd.DataFrame(columns=cols) if df_existing_timestamps is None else df_existing_timestamps
    for (participant_id, session_id) in product(SUBJECTS, SESSIONS):
        print(participant_id, session_id)
        # check if there's an entry for this participant and session already
        if not df.loc[(df['participant_id'] == participant_id) & (df['session_id'] == session_id)].empty:
            continue
        data_apdm = get_apdm_data_cut(participant_id, session_id, condition="RunTM")
        if not data_apdm:
            continue
        cm = get_cut_marks(data_apdm, sensor_location='left_foot')
        t = data_apdm.get('left_foot').get('timestamp')
        timestamp_start = t[cm[0]]
        timestamp_end = t[cm[1]]
        new_row = pd.DataFrame({"participant_id": participant_id,
                                "session_id": session_id,
                                "timestamp_start": timestamp_start,
                                "timestamp_end": timestamp_end
                                }, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)
        save_cut_marks_final(df)


def make_final_cut_plots(df_timestamps: pd.DataFrame):
    path_plot = PATH_DATA_ROOT.joinpath('plots', "sync_cut_check")
    path_plot.mkdir(exist_ok=True, parents=True)
    for (participant_id, session_id) in product(SUBJECTS, SESSIONS):
        print(participant_id, session_id)
        # check if there's an entry for this participant and session already
        row = df_timestamps.loc[(df_timestamps['participant_id'] == participant_id) & (df_timestamps['session_id'] == session_id)]
        if row.empty:
            continue
        t_start = row["timestamp_start"].values[0]
        t_end = row["timestamp_end"].values[0]
        data_apdm = get_apdm_data_cut(participant_id, session_id, condition="RunTM")
        data_knb = get_knb_data(participant_id, session_id)
        if not all([data_apdm, data_knb]):
            if participant_id in SKIP.keys():
                if session_id in SKIP[participant_id]:
                    print(f'Skipping {participant_id} - {session_id}')
                    continue
        sensor_names = ['left_foot', 'left_tibia', 'pelvis', 'sternum']

        for sensor_name in sensor_names:
            filename = f"{participant_id}_{session_id}_{sensor_name}.png"
            path_file = path_plot.joinpath(filename)
            if path_file.exists():
                print(f'Plot {path_file.stem} already exists. Skipping...')
                continue
            plt.close(plt.gcf())
            fig, ax = plt.subplots(figsize=(36, 18))
            t_apdm = data_apdm[sensor_name]["timestamp"]
            gyr_x_apdm = data_apdm[sensor_name]['gyr'][:, 0]
            gyr_x_apdm = gyr_x_apdm * (180.0 / 3.141592653589793)  # convert to deg/s
            t_knb = data_knb[sensor_name]["timestamp"]
            gyr_x_knb = data_knb[sensor_name]['gyr'][:, 0]

            i_start_knb = np.where(t_knb > t_start)[0][0]
            i_end_knb = np.where(t_knb < t_end)[0][-1]

            i_start_apdm = np.where(t_apdm > t_start)[0][0]
            i_end_apdm = np.where(t_apdm < t_end)[0][-1]

            ax.plot(t_apdm[i_start_apdm:i_end_apdm], gyr_x_apdm[i_start_apdm:i_end_apdm])
            ax.plot(t_knb[i_start_knb:i_end_knb], gyr_x_knb[i_start_knb:i_end_knb], linestyle="--")
            # ax.axvline(t_start, color="k", linestyle="--")
            # ax.axvline(t_end, color="k", linestyle="--")
            title = f"{participant_id} - {session_id} - {sensor_name}"
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(path_file)
            plt.close(fig)


def cut_files_final(df_timestamps: pd.DataFrame):
    path_out = PATH_DATA_ROOT.joinpath('sync_cut')

    for (participant_id, session_id) in product(SUBJECTS, SESSIONS):
        print(f"Cutting {participant_id} - {session_id}...")
        # check if there's an entry for this participant and session already
        row = df_timestamps.loc[(df_timestamps['participant_id'] == participant_id) & (df_timestamps['session_id'] == session_id)]
        if row.empty:
            continue
        t_start = row["timestamp_start"].values[0]
        t_end = row["timestamp_end"].values[0]
        data_apdm = get_apdm_data_cut(participant_id, session_id, condition="RunTM")
        data_knb = get_knb_data(participant_id, session_id)
        if not all([data_apdm, data_knb]):
            if participant_id in SKIP.keys():
                if session_id in SKIP[participant_id]:
                    print(f'Skipping {participant_id} - {session_id}')
                    continue
        data_knb_cut = cut_sensor_data(data_knb, t_start, t_end)
        data_apdm_cut = cut_sensor_data(data_apdm, t_start, t_end)

        path_out_participant = path_out.joinpath(participant_id, session_id)
        path_out_participant.mkdir(parents=True, exist_ok=True)
        path_knb_cut_file = path_out_participant.joinpath(f"{participant_id}_{session_id}_RunTM_KiNetBlue.hdf5")
        path_apdm_cut_file = path_out_participant.joinpath(f"{participant_id}_{session_id}_RunTM_APDM.hdf5")
        save_dict_to_hdf5(data_knb_cut, path_knb_cut_file)
        save_dict_to_hdf5(data_apdm_cut, path_apdm_cut_file)


def cut_sensor_data(data: dict, t_start: int, t_end: int) -> dict:
    sensor_names = ['left_foot', 'left_tibia', 'pelvis', 'sternum']
    assert sensor_names == list(data.keys())  # make sure the keys are as expected (i.e. sensor names)
    data_cut = dict().fromkeys(data.keys())
    for sensor_name in data_cut.keys():
        t = data[sensor_name]["timestamp"]
        i_start = np.where(t > t_start)[0][0]
        i_end = np.where(t < t_end)[0][-1]
        # print(f'Cutting {sensor_name}: {i_start} to {i_end}. Total length: {len(t)}')
        data_cut_sensor = dict()
        data_cut_sensor['acc'] = data[sensor_name]["acc"][i_start:i_end]
        data_cut_sensor['gyr'] = data[sensor_name]["gyr"][i_start:i_end]
        data_cut_sensor['timestamp'] = data[sensor_name]["timestamp"][i_start:i_end]
        data_cut[sensor_name] = data_cut_sensor
    return data_cut


def make_check_plots_final():
    path_root = PATH_DATA_ROOT.joinpath('sync_cut')
    path_plot = PATH_DATA_ROOT.joinpath('plots', "final_cut_check")
    path_plot.mkdir(exist_ok=True, parents=True)
    for p_id, s_id in product(SUBJECTS, SESSIONS):
        filename_knb = f"{p_id}_{s_id}_RunTM_KiNetBlue.hdf5"
        path_knb_file = path_root.joinpath(p_id, s_id, filename_knb)
        if not path_knb_file.exists():
            continue
        filename_apdm = f"{p_id}_{s_id}_RunTM_APDM.hdf5"
        path_apdm_file = path_root.joinpath(p_id, s_id, filename_apdm)
        if not path_apdm_file.exists():
            continue
        print(f"Making plots for {p_id} - {s_id}...")
        path_plot_participant = path_plot.joinpath(p_id)
        path_plot_participant.mkdir(exist_ok=True, parents=True)
        data_knb = load_dict_from_hdf5(path_knb_file)
        data_apdm = load_dict_from_hdf5(path_apdm_file)
        for sensor_name in SENSOR_LOCATIONS:
            fig, ax = plt.subplots(figsize=(36, 18))
            t_apdm = data_apdm[sensor_name]["timestamp"]
            gyr_x_apdm = data_apdm[sensor_name]['gyr'][:, 0]
            gyr_x_apdm = gyr_x_apdm * (180.0 / 3.141592653589793)
            t_knb = data_knb[sensor_name]["timestamp"]
            gyr_x_knb = data_knb[sensor_name]['gyr'][:, 0]
            ax.plot(t_apdm, gyr_x_apdm)
            ax.plot(t_knb, gyr_x_knb, linestyle="--")
            title = f"{p_id} - {s_id} - {sensor_name}"
            ax.set_title(title)
            plt.tight_layout()
            filename = f"{p_id}_{s_id}_{sensor_name}.png"
            path_file = path_plot_participant.joinpath(filename)
            plt.savefig(path_file)
            plt.close(fig)


def main():
    df_timestamps = load_cut_marks_final()
    # get_timestamps(df_timestamps)
    # make_final_cut_plots(df_timestamps)
    # cut_files_final(df_timestamps)
    make_check_plots_final()


if __name__ == '__main__':
    main()
