from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
PATH_DATA_ROOT = Path(os.getenv("PATH_DATA_ROOT"))
PATH_APDM_RAW = PATH_DATA_ROOT.joinpath('APDM', 'raw')
PATH_KINETBLUE_RAW = PATH_DATA_ROOT.joinpath('KiNetBlue', 'raw')
PATH_APDM_CUT = PATH_DATA_ROOT.joinpath('APDM', 'cut')
PATH_KINETBLUE_CUT = PATH_DATA_ROOT.joinpath('KiNetBlue', 'cut')

SUBJECTS = [f'S{str(s).zfill(2)}' for s in range(1, 26)]
SESSIONS = [f'S{str(s)}' for s in range(1, 4)]
SENSOR_LOCATIONS = ['pelvis', 'sternum']
