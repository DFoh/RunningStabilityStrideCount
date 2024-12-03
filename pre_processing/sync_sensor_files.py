from itertools import product

import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    for subject, session, sensor_location in product(SUBJECTS, SESSIONS, SENSOR_LOCATIONS):
        print(f'Processing {subject} - {session} - {sensor_location}...')
        data_apdm = get_apdm_data(subject, session, sensor_location)
        data_kinetblue = get_kinetblue_data(subject, session, sensor_location)

        time_apdm = data_apdm['Time'] - data_apdm['Time'][0]
        acc_apdm = data_apdm['Accelerometer']
        time_kinetblue = data_kinetblue['timestamp'] - data_kinetblue['timestamp'][0]
        acc_kinetblue = data_kinetblue['acc'] * 9.81
        fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        ax[0].plot(time_apdm, acc_apdm)
        ax[1].plot(time_kinetblue, acc_kinetblue)
        plt.show()
        pass
