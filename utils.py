import os
from pathlib import Path

from dotenv import load_dotenv

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
