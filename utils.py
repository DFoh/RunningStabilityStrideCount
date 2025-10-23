import os
from pathlib import Path

from dotenv import load_dotenv
from labtools.systems.apdm import get_apdm_sensor_data_by_location
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


def get_apdm_data(subject: str, session: str, sensor_location: str = None):
    session_path = PATH_APDM_CUT
    if [sub for sub in session_path.iterdir() if sub.name == 'subjects']:
        session_path = PATH_APDM_RAW.joinpath('subjects')
    session_path = session_path.joinpath(subject).joinpath(session)
    if not session_path.exists():
        return None
    session_file = [f for f in session_path.iterdir() if "RunTM" in f.name]
    if len(session_file) != 1:
        return None
    data = load_dict_from_hdf5(session_file[0])
    if sensor_location is None:
        if 'data' in data.keys():
            data = data['data']
        return data
    sensor_location = SENSOR_LOCATIONS_APDM[sensor_location]
    return get_apdm_sensor_data_by_location(data, sensor_location)
