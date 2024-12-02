from utils import *
from labtools.utils.file_handling import load_dict_from_hdf5, save_dict_to_hdf5

from itertools import product


def get_kinetblue_data(subject: str, session: str, sensor_location: str):
    session_path = PATH_KINETBLUE_RAW.joinpath('subjects').joinpath(subject).joinpath(session)
    if not session_path.exists():
        return None
    treadmill_file = [print(f) for f in session_path.iterdir()]
    if len(treadmill_file) != 1:
        return None
    data = load_dict_from_hdf5(treadmill_file[0])
    foo = 1
    pass


def get_apdm_data(subject: str, session: str, sensor_location: str):
    session_path = PATH_APDM_RAW.joinpath('subjects').joinpath(subject).joinpath(session)
    if not session_path.exists():
        return None
    session_file = [f for f in session_path.iterdir()]
    if len(session_file) != 1:
        return None
    data = load_dict_from_hdf5(session_file[0])
    sensor_data = data['data'][sensor_location]
    foo = 1
    pass


if __name__ == '__main__':
    for subject, session, sensor_location in product(SUBJECTS, SESSIONS, SENSOR_LOCATIONS):
        print(f'Processing {subject} - {session} - {sensor_location}...')
        get_apdm_data(subject, session, sensor_location)
        foo = 1
        pass
