import warnings
from itertools import product

import matplotlib.pyplot as plt
from labtools.utils.hdf5 import load_dict_from_hdf5

from utils import *


def get_kinetblue_synced_data(participant_id, session_id):
    path_sync_files = PATH_KINETBLUE_RAW.as_posix().replace('raw', 'sync')
    path_knb_file = Path(path_sync_files).joinpath(participant_id, session_id)
    files_knb = list(path_knb_file.glob('*.hdf5'))
    file_knb = [f for f in files_knb if 'RunTM_sync' in f.name]
    if len(file_knb) != 1:
        return None

    data_knb = load_dict_from_hdf5(file_knb[0])
    return data_knb

if __name__ == '__main__':
    path_plot_root = PATH_DATA_ROOT.joinpath('plots', 'timestamp_check')
    path_plot_root.mkdir(parents=True, exist_ok=True)
    for p_id, session_id in product(SUBJECTS, SESSIONS):
        print(f'Loading data for {p_id} - {session_id}...')
        data_apdm = get_apdm_data(p_id, session_id)
        if data_apdm is None:
            warnings.warn(f'No cut APDM data for {p_id} - {session_id}. Skipping...')
            continue

        data_knb = get_kinetblue_synced_data(p_id, session_id)
        if data_knb is None:
            warnings.warn(f'No synced KiNetBlue data for {p_id} - {session_id}. Skipping...')
            continue


        gyr_x_apdm = data_apdm['left_foot']['gyr'][:, 0]  # rad/s
        gyr_x_apdm = gyr_x_apdm * (180.0 / 3.141592653589793)  # convert to deg/s
        t_apdm = data_apdm['left_foot']['timestamp']

        gyr_knb = data_knb['left_foot']['gyr'][:, 0]  # deg/s
        t_knb = data_knb['left_foot']['timestamp']

        t_min = t_knb.min()
        t_max = t_knb.max()
        t_mid = (t_min + t_max) / 2
        t_mid_plus_1_s = t_mid + 1_000_000

        plt.figure(figsize=(12, 8))
        plt.plot(t_apdm, gyr_x_apdm, label='APDM')
        plt.plot(t_knb, gyr_knb, label='KiNetBlue', linestyle='--')
        plt.xlabel('Time (s)')
        plt.gca().set_xlim([t_mid, t_mid_plus_1_s])
        title = f'Gyroscope X-axis - {p_id} - {session_id}'
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        filename = f'{p_id}-{session_id}.png'
        path_plot_out = path_plot_root.joinpath(filename)
        plt.savefig(path_plot_out)
