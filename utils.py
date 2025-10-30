import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from labtools.systems.apdm.apdm import get_apdm_sensor_data_by_location
from labtools.utils.hdf5 import load_dict_from_hdf5

load_dotenv()
PATH_DATA_ROOT = Path(os.getenv("PATH_DATA_ROOT"))
PATH_APDM_RAW = PATH_DATA_ROOT.joinpath('raw', 'APDM')
PATH_KINETBLUE_RAW = PATH_DATA_ROOT.joinpath('raw', 'KiNetBlue')
PATH_APDM_CUT = PATH_DATA_ROOT.joinpath('cut', 'APDM')
PATH_KINETBLUE_CUT = PATH_DATA_ROOT.joinpath('KiNetBlue', 'cut')

SUBJECTS = [f'S{str(s).zfill(2)}' for s in range(1, 26)]
SESSIONS = [f'S{str(s)}' for s in range(1, 4)]
SENSOR_LOCATIONS = ['pelvis', 'sternum', 'left_tibia', 'left_foot']
SENSOR_LOCATIONS_APDM = {
    'pelvis': 'Lumbar',
    'sternum': 'Sternum',
    'left_tibia': 'Left Tibia',
    'left_foot': 'Left Foot',
}

SKIP = {'S08': ['S1'],  # KiNetBlue file corrupt
        'S23': ['S3'],  # KiNetBlue file corrupt
        }


def get_apdm_data_cut(subject: str, session: str, sensor_location: str = None, condition: str = ''):
    return get_apdm_data(subject=subject, session=session, path_data_root=PATH_APDM_CUT, sensor_location=sensor_location, condition=condition)


def get_apdm_data_raw(subject: str, session: str, sensor_location: str = None):
    return get_apdm_data(subject=subject, session=session, path_data_root=PATH_APDM_RAW, sensor_location=sensor_location)


def get_apdm_data(subject: str, session: str, path_data_root: Path, condition: str = '', sensor_location: str = None):
    path_root = path_data_root
    session_path = path_root.joinpath(subject).joinpath(session)
    if not session_path.exists():
        return None
    files = list(session_path.glob("*.hdf5"))
    session_file = [f for f in files if condition in f.name]
    if len(session_file) != 1:
        warnings.warn("Unexpected number of APDM files found.")
        return None
    data = load_dict_from_hdf5(session_file[0])
    if sensor_location is None:
        if 'data' in data.keys():
            data = data['data']
        return data
    sensor_location = SENSOR_LOCATIONS_APDM[sensor_location]
    return get_apdm_sensor_data_by_location(data, sensor_location)


def get_apdm_sensor_data_by_location_hardcoded(hdf5_data: dict, location: str) -> dict:
    location_map = {'XI-013063': 'Lumbar', 'XI-013303': 'Sternum', 'XI-013311': 'Left Foot', 'XI-013331': 'Left Tibia'}

    if location not in location_map.values():
        raise ValueError(f'Location {location} not found in the sensor data.')
    sensor_data = dict()
    for sensor_id, loc in location_map.items():
        if loc == location:
            sensor_data = hdf5_data['Sensors'][sensor_id]
    return sensor_data
