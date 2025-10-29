import warnings
from itertools import product

import matplotlib.backend_bases as back
import matplotlib.pyplot as plt
import numpy as np
from labtools.utils.hdf5 import save_dict_to_hdf5
from scipy.signal import find_peaks

from utils import *


def get_kinetblue_data(subject: str, session: str, sensor_location: str = None):
    session_path = PATH_KINETBLUE_RAW
    if [sub for sub in session_path.iterdir() if sub.name == 'subjects']:
        session_path.joinpath('subjects')
    session_path = session_path.joinpath(subject).joinpath(session)
    if not session_path.exists():
        return None
    treadmill_file = [f for f in session_path.iterdir() if 'RunTM' in f.name]
    if len(treadmill_file) != 1:
        return None
    data = load_dict_from_hdf5(treadmill_file[0])['data']
    if sensor_location is None:
        return data
    if sensor_location not in data.keys():
        raise ValueError(f'Location {sensor_location} not found in the sensor data.')
    return data[sensor_location]


def get_peak_range(signal, time_ms, n_seconds=30):
    # get the range from ginput in which the three knocks are present in the signal
    sample_rate_hz = 1E6 / np.mean(np.diff(time_ms))
    n_samples = int(n_seconds * sample_rate_hz)
    signal_range = signal[:n_samples]
    time_range = time_ms[:n_samples]
    fig, ax = plt.subplots()
    ax.plot(time_ms, signal)
    ax.set_xlim([time_range[0], time_range[-1]])
    fig.suptitle('Select the range of the signal where the three knocks are present.')
    pts = fig.ginput(2, timeout=0, mouse_add=back.MouseButton.RIGHT, mouse_pop=None)
    t1 = int(pts[0][0])
    t2 = int(pts[1][0])
    ax.axvline(t1, color='r')
    ax.axvline(t2, color='r')
    plt.close(fig)
    x1 = np.argmin(np.abs(time_ms - t1))
    x2 = np.argmin(np.abs(time_ms - t2))

    return x1, x2


def get_peaks(data: dict, system: str):
    if system == 'apdm':
        acc = data.get('Accelerometer')
        if acc is None:
            acc = data.get('acc')
        acc_res = np.linalg.norm(acc, axis=1)
        time_ms = data.get('Time')
        if time_ms is None:
            time_ms = data.get('timestamp')
    elif system == 'kinetblue':
        acc = data['acc'] * 9.81
        acc_res = np.linalg.norm(acc, axis=1)
        time_ms = data['timestamp']
    else:
        raise ValueError('System not recognized.')

    x1, x2 = get_peak_range(acc_res, time_ms)
    sample_rate_hz = int(1E6 / np.mean(np.diff(time_ms)))
    signal_range = acc_res[x1:x2]

    signal_max = np.max(signal_range)
    peaks, _ = find_peaks(signal_range, height=signal_max / 4, distance=sample_rate_hz / 4)
    peaks_refined = peaks[np.argsort(signal_range[peaks])[-3:]]
    # pplot results
    # fig, ax = plt.subplots()
    # ax.plot(signal_range)
    # for peak in peaks:
    #     ax.scatter(peak, signal_range[peak], c='r')
    # filter to get the three highest peaks
    # for peak in peaks_refined:
    #     ax.scatter(peak, signal_range[peak], c='g')
    # plt.show()
    if len(peaks_refined) != 3:
        raise ValueError('Three peaks were not found in the signal.')

    return peaks_refined + x1


def get_time_delta(peaks_apdm, peaks_kinetblue, time_apdm, time_kinetblue) -> float:
    t1_apdm = time_apdm[peaks_apdm[0]]
    t2_apdm = time_apdm[peaks_apdm[1]]
    t3_apdm = time_apdm[peaks_apdm[2]]
    dt1_apdm = t2_apdm - t1_apdm
    dt2_apdm = t3_apdm - t2_apdm
    t_mean_apdm = np.mean([t1_apdm, t2_apdm, t3_apdm])
    t1_kinetblue = time_kinetblue[peaks_kinetblue[0]]
    t2_kinetblue = time_kinetblue[peaks_kinetblue[1]]
    t3_kinetblue = time_kinetblue[peaks_kinetblue[2]]
    t_mean_kinetblue = np.mean([t1_kinetblue, t2_kinetblue, t3_kinetblue])
    dt1_kinetblue = t2_kinetblue - t1_kinetblue
    dt2_kinetblue = t3_kinetblue - t2_kinetblue

    dt_systems_1 = t1_apdm - t1_kinetblue
    dt_systems_2 = t2_apdm - t2_kinetblue
    dt_systems_3 = t3_apdm - t3_kinetblue
    dt_systems_mean = t_mean_apdm - t_mean_kinetblue
    return dt_systems_mean


def process_sensor_location(data_apdm: dict, data_kinetblue: dict, sensor_location: str):
    loc_apdm = SENSOR_LOCATIONS_APDM.get(sensor_location)
    if loc_apdm is None:
        raise ValueError(f'Location {sensor_location} not found in the APDM sensor data.')
    if sensor_location not in data_apdm.keys():
        data_apdm_loc = get_apdm_sensor_data_by_location(data_apdm, loc_apdm)
    else:
        data_apdm_loc = data_apdm[sensor_location]
    # data_apdm_loc = get_apdm_sensor_data_by_location(data_apdm, sensor_location)
    data_kinetblue_loc = data_kinetblue[sensor_location]

    t_orig = data_kinetblue_loc['timestamp'].copy()
    print(f'{t_orig[0]=}')

    # stage 1 - get the time delta based on the three knocks

    peaks_apdm = get_peaks(data_apdm_loc, 'apdm')
    peaks_kinetblue = get_peaks(data_kinetblue_loc, 'kinetblue')
    dt_systems = get_time_delta(peaks_apdm, peaks_kinetblue, data_apdm_loc['timestamp'],
                                data_kinetblue_loc['timestamp'])
    # adjust kinetblue timestampts
    data_kinetblue_loc['timestamp'] = data_kinetblue_loc['timestamp'] + dt_systems
    print('# # # # # # # # #')
    print(f'Time delta - stage 1: {dt_systems / 1E6}seconds')
    print('# # # # # # # # #')

    t_stage_1 = data_kinetblue_loc['timestamp'].copy()
    print(f'{t_orig[0]=}')
    print(f'{t_stage_1[0]=}')

    # Plot the time-offset signals
    time_apdm = data_apdm_loc.get('Time')
    if time_apdm is None:
        time_apdm = data_apdm_loc.get('timestamp')
    time_kinetblue = data_kinetblue_loc['timestamp']
    gyro_apdm = data_apdm_loc.get('Gyroscope')
    if gyro_apdm is None:
        gyro_apdm = data_apdm_loc.get('gyr')
    gyro_apdm = gyro_apdm[:, 0] * np.rad2deg(1)
    gyro_kinetblue = data_kinetblue_loc['gyr'][:, 0]
    # fig, ax = plt.subplots()
    # ax.plot(time_apdm, gyro_apdm, 'k')
    # ax.plot(time_kinetblue, gyro_kinetblue, 'r--')
    # plt.show()

    # stage 2 - synchronize the signals through convolution
    new_sample_rate = 2000  # Hz

    t_min = max(time_apdm[0], time_kinetblue[0])
    t_max = min(time_apdm[-1], time_kinetblue[-1])

    time_resampled = np.arange(t_min, t_max, 1E6 / new_sample_rate)
    gyro_apdm_resampled = np.interp(time_resampled, time_apdm, gyro_apdm)
    gyro_kinetblue_resampled = np.interp(time_resampled, time_kinetblue, gyro_kinetblue)

    optimal_lags = []
    for i in range(10):
        start = i * 10000 + 20000
        end = i * 10000 + 30000

        t_r = time_resampled[start:end]
        g_k = gyro_kinetblue_resampled[start:end]
        g_a = gyro_apdm_resampled[start:end]
        # Perform cross-correlation over the band of interest (+/-20 samples) to find the optimal lag
        lag_range = 300  # +/- 20 samples
        lags = np.arange(-lag_range, lag_range + 1)
        cross_corr = np.array([np.sum(g_k * np.roll(g_a, lag)) for lag in lags])

        # Find the optimal lag
        optimal_lag_index = np.argmax(cross_corr)
        optimal_lag = lags[optimal_lag_index]
        optimal_lags.append(optimal_lag)

        print(f'Optimal lag for synchronization: {optimal_lag} samples')
    optimal_lag = np.median(optimal_lags)
    if optimal_lag > 50:
        warnings.warn('Optimal lag is greater than 50 samples. Check the synchronization.')
    optimal_timeshift = optimal_lag / new_sample_rate * 1E6
    # Adjust the timestamps
    data_kinetblue_loc['timestamp'] = data_kinetblue_loc['timestamp'] - optimal_timeshift

    t_stage_2 = data_kinetblue_loc['timestamp'].copy()

    # Plot the time-offset signals
    time_apdm = data_apdm_loc.get('Time')
    if time_apdm is None:
        time_apdm = data_apdm_loc.get('timestamp')
    time_kinetblue = data_kinetblue_loc['timestamp']
    gyro_apdm = data_apdm_loc.get('Gyroscope')
    if gyro_apdm is None:
        gyro_apdm = data_apdm_loc.get('gyr')
    gyro_apdm = gyro_apdm[:, 0] * np.rad2deg(1)
    gyro_kinetblue = data_kinetblue_loc['gyr'][:, 0]
    fig, ax = plt.subplots()
    ax.plot(time_apdm, gyro_apdm, 'k', label='APDM')
    # ax.plot(t_orig, gyro_kinetblue, 'r--', label='KiNetBlue orig')
    ax.plot(time_kinetblue, gyro_kinetblue, 'g--', label='KiNetBlue stage 2')
    plt.legend()
    plt.show()
    return data_kinetblue_loc


if __name__ == '__main__':

    for subject, session in product(SUBJECTS, SESSIONS):
        print(f'Processing {subject} - {session}')
        if subject in SKIP.keys():
            if session in SKIP[subject]:
                print(f'Skipping {subject} - {session}')
                continue
        # Check if file already exists
        path_sync_out = PATH_KINETBLUE_RAW.as_posix().replace('raw', 'sync')
        path_sync_file_out = Path(path_sync_out).joinpath(subject).joinpath(session)
        path_sync_file_out.mkdir(parents=True, exist_ok=True)
        filename = f'{subject}_{session}_RunTM_sync.hdf5'
        path_file_out = path_sync_file_out.joinpath(filename)
        if path_file_out.exists():
            print(f'File {path_file_out} already exists. Skipping...')
            continue
        # load data
        # Todo: handle missing data
        data_apdm = get_apdm_data_cut(subject, session, condition="RunTM")
        data_kinetblue = get_kinetblue_data(subject, session)
        for location in SENSOR_LOCATIONS:
            try:
                data_kinetblue_loc = process_sensor_location(data_apdm, data_kinetblue, location)
            except Exception as e:
                print(f'Error processing {location}: {e}')
                data_kinetblue_loc = None
                continue
            data_kinetblue[location] = data_kinetblue_loc
        save_dict_to_hdf5(data_kinetblue, path_file_out)
