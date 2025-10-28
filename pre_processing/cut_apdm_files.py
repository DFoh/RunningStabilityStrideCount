import json
from itertools import product

from labtools.utils.cutting_tool import InteractiveCuttingTool
from labtools.utils.hdf5 import load_dict_from_hdf5, save_dict_to_hdf5

from utils import *

KNOWN_CUT_MARKS_FILE = Path(PATH_DATA_ROOT).joinpath('known_cut_marks.json')
if KNOWN_CUT_MARKS_FILE.exists():
    with open(KNOWN_CUT_MARKS_FILE, 'r') as file:
        KNOWN_CUT_MARKS = json.load(file)
else:
    KNOWN_CUT_MARKS = dict()


def save_cut_marks(subject: str, session: str, cut_marks: list):
    if subject not in KNOWN_CUT_MARKS:
        KNOWN_CUT_MARKS[subject] = dict()
    KNOWN_CUT_MARKS[subject][session] = cut_marks
    with open(KNOWN_CUT_MARKS_FILE, 'w') as file:
        json.dump(KNOWN_CUT_MARKS, file)


def get_cut_marks(data: dict, sensor_location: str = None) -> list:
    cutter = InteractiveCuttingTool(num_pieces=1,
                                    mode="start_stop")
    cut_marks = list()
    data = data['Sensors'] if "Sensors" in data.keys() else data
    for sensor_id, sensor_data in data.items():
        if sensor_location and sensor_location != sensor_id:
            continue
        acc = sensor_data['Accelerometer'] if 'Accelerometer' in sensor_data.keys() else sensor_data['acc']
        cm = cutter.get_cut_marks(display_data=acc)
        cut_marks.extend(cm)
    return [min(cut_marks), max(cut_marks)]


def cut_dataset(data: dict, key: str, cut_marks: list) -> dict:
    data_out = data[key]
    if cut_marks[0] < 0:
        raise ValueError(f'Cut mark {cut_marks[0]} is less than 0.')
    if cut_marks[1] > len(data[key]):
        raise ValueError(f'Cut mark {cut_marks[1]} is greater than the length of the data {len(data[key])}.')
    return data_out[cut_marks[0]:cut_marks[1]]


def cut_data(data: dict, cut_marks: list) -> dict:
    data_out = dict().fromkeys(data.keys())
    data_out['Sensors'] = dict()
    for sensor_id, sensor_data in data['Sensors'].items():
        acc_cut = cut_dataset(sensor_data, 'Accelerometer', cut_marks=cut_marks)
        gyro_cut = cut_dataset(sensor_data, 'Gyroscope', cut_marks=cut_marks)
        timestamp_cut = cut_dataset(sensor_data, 'Time', cut_marks=cut_marks)
        magnetometer_cut = cut_dataset(sensor_data, 'Magnetometer', cut_marks=cut_marks)
        data_out['Sensors'][sensor_id] = {'Accelerometer': acc_cut,
                                          'Gyroscope': gyro_cut,
                                          'Magnetometer': magnetometer_cut,
                                          'Time': timestamp_cut,
                                          'Configuration': sensor_data['Configuration']}
    return data_out


def get_cut_apdm_data(subject: str, session: str):
    session_path = PATH_APDM_RAW
    if [sub for sub in PATH_APDM_RAW.iterdir() if sub.name == 'subjects']:
        session_path = PATH_APDM_RAW.joinpath('subjects')
    session_path = session_path.joinpath(subject).joinpath(session)
    if not session_path.exists():
        return None
    session_file = [f for f in session_path.iterdir()]
    if len(session_file) != 1:
        return None
    data = load_dict_from_hdf5(session_file[0])
    if KNOWN_CUT_MARKS.get(subject, {}).get(session):
        cut_marks = KNOWN_CUT_MARKS[subject][session]
        return cut_data(data, cut_marks)
    cut_marks = get_cut_marks(data)
    save_cut_marks(subject, session, cut_marks)
    return cut_data(data, cut_marks)


if __name__ == '__main__':
    for subject, session in product(SUBJECTS, SESSIONS):
        print(f'Processing {subject} - {session} ...')
        path_out = PATH_APDM_CUT.joinpath(subject).joinpath(session)
        path_out.mkdir(parents=True, exist_ok=True)
        file_name = f'{subject}_{session}_RunTM.hdf5'
        path_out = path_out.joinpath(file_name)
        if path_out.exists():
            print(f'File {path_out} already exists. Skipping...')
            continue

        data_cut = get_cut_apdm_data(subject, session)

        save_dict_to_hdf5(data_cut, path_out)
