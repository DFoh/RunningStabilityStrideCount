from itertools import product

import matplotlib.pyplot as plt
import matplotlib.backend_bases as back

import numpy as np
from scipy.signal import find_peaks

from labtools.systems.apdm import get_apdm_sensor_data_by_location
from labtools.utils.hdf5 import load_dict_from_hdf5

from utils import *


def get_kinetblue_data(subject: str, session: str, sensor_location: str):
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
    if sensor_location not in data.keys():
        raise ValueError(f'Location {sensor_location} not found in the sensor data.')
    return data[sensor_location]


def get_apdm_data(subject: str, session: str, sensor_location: str):
    sensor_location = SENSOR_LOCATIONS_APDM[sensor_location]
    session_path = PATH_APDM_CUT
    if [sub for sub in session_path.iterdir() if sub.name == 'subjects']:
        session_path = PATH_APDM_RAW.joinpath('subjects')
    session_path = session_path.joinpath(subject).joinpath(session)
    if not session_path.exists():
        return None
    session_file = [f for f in session_path.iterdir()]
    if len(session_file) != 1:
        return None
    data = load_dict_from_hdf5(session_file[0])
    return get_apdm_sensor_data_by_location(data, sensor_location)


def get_peak_range(signal, time_ms, n_seconds=30):
    # get the range from ginput in which the three knocks are present in the signal
    sample_rate_hz = 1E6 / np.mean(np.diff(time_ms))
    n_samples = int(n_seconds * sample_rate_hz)
    signal_range = signal[:n_samples]
    time_range = time_ms[:n_samples]
    fig, ax = plt.subplots()
    ax.plot(time_range, signal_range)
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
        acc = data['Accelerometer']
        acc_res = np.linalg.norm(acc, axis=1)
        time_ms = (data['Time'])
    elif system == 'kinetblue':
        acc = data['acc'] * 9.81
        acc_res = np.linalg.norm(acc, axis=1)
        time_ms = (data['timestamp'])
    else:
        raise ValueError('System not recognized.')

    x1, x2 = get_peak_range(acc_res, time_ms)
    sample_rate_hz = int(1E6 / np.mean(np.diff(time_ms)))
    signal_range = acc_res[x1:x2]
    fig, ax = plt.subplots()
    ax.plot(signal_range)
    signal_max = np.max(signal_range)
    peaks, _ = find_peaks(signal_range, height=signal_max / 2, distance=sample_rate_hz / 4)
    for peak in peaks:
        ax.scatter(peak, signal_range[peak], c='r')
    plt.show()
    if len(peaks) != 3:
        raise ValueError('Three peaks were not found in the signal.')

    return peaks + x1


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


if __name__ == '__main__':
    for subject, session, sensor_location in product(SUBJECTS, SESSIONS, SENSOR_LOCATIONS):
        print(f'Processing {subject} - {session} - {sensor_location}...')
        data_apdm = get_apdm_data(subject, session, sensor_location)
        data_kinetblue = get_kinetblue_data(subject, session, sensor_location)

        t_orig = data_kinetblue['timestamp'].copy()
        print(f'{t_orig[0]=}')

        # stage 1 - get the time delta based on the three knocks

        peaks_apdm = get_peaks(data_apdm, 'apdm')
        peaks_kinetblue = get_peaks(data_kinetblue, 'kinetblue')
        dt_systems = get_time_delta(peaks_apdm, peaks_kinetblue, data_apdm['Time'], data_kinetblue['timestamp'])
        # adjust kinetblue timestampts
        data_kinetblue['timestamp'] = data_kinetblue['timestamp'] + dt_systems
        print('# # # # # # # # #')
        print(f'Time delta - stage 1: {dt_systems / 1E6}seconds')
        print('# # # # # # # # #')

        t_stage_1 = data_kinetblue['timestamp'].copy()
        print(f'{t_orig[0]=}')
        print(f'{t_stage_1[0]=}')

        # Plot the time-offset signals
        time_apdm = data_apdm['Time']
        time_kinetblue = data_kinetblue['timestamp']
        gyro_apdm = data_apdm['Gyroscope'][:, 0] * np.rad2deg(1)
        gyro_kinetblue = data_kinetblue['gyr'][:, 0]
        fig, ax = plt.subplots()
        ax.plot(time_apdm, gyro_apdm, 'k')
        ax.plot(time_kinetblue, gyro_kinetblue, 'r--')
        plt.show()

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
            lag_range = 30  # +/- 20 samples
            lags = np.arange(-lag_range, lag_range + 1)
            cross_corr = np.array([np.sum(g_k * np.roll(g_a, lag)) for lag in lags])

            # Find the optimal lag
            optimal_lag_index = np.argmax(cross_corr)
            optimal_lag = lags[optimal_lag_index]
            optimal_lags.append(optimal_lag)

            print(f'Optimal lag for synchronization: {optimal_lag} samples')
        optimal_lag = np.median(optimal_lags)
        optimal_timeshift = optimal_lag / new_sample_rate * 1E6
        # Adjust the timestamps
        data_kinetblue['timestamp'] = data_kinetblue['timestamp'] - optimal_timeshift

        t_stage_2 = data_kinetblue['timestamp'].copy()
        print(f'{t_orig[0]=}')
        print(f'{t_stage_1[0]=}')
        print(f'{t_stage_2[0]=}')

        # Plot the time-offset signals
        time_apdm = data_apdm['Time']
        time_kinetblue = data_kinetblue['timestamp']
        gyro_apdm = data_apdm['Gyroscope'][:, 0] * np.rad2deg(1)
        gyro_kinetblue = data_kinetblue['gyr'][:, 0]
        fig, ax = plt.subplots()
        ax.plot(time_apdm, gyro_apdm, 'k', label='APDM')
        # ax.plot(t_orig, gyro_kinetblue, 'r--', label='KiNetBlue orig')
        ax.plot(t_stage_1, gyro_kinetblue, 'b--', label='KiNetBlue stage 1')
        ax.plot(t_stage_2, gyro_kinetblue, 'g--', label='KiNetBlue stage 2')
        plt.legend()
        plt.show()
