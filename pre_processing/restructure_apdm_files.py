# script to get the APDM files in the internal structure by location and "acc", "gyr", "timestamp" keys
import warnings

from labtools.utils.hdf5 import save_dict_to_hdf5

from utils import *

if __name__ == '__main__':
    path_raw = PATH_APDM_CUT
    for path_participant in path_raw.iterdir():
        p_id = path_participant.stem
        if p_id != "S05":
            continue
        for path_session in path_participant.iterdir():
            session_id = path_session.stem
            print(f'Processing {p_id} - {session_id}...')
            # data = get_apdm_data_raw(p_id, session_id)
            data = get_apdm_data_cut(p_id, session_id, condition="RunTM")
            filename = f'{p_id}_{session_id}_RunTM.hdf5'
            filepath = path_session.joinpath(filename)
            if data is None:
                warnings.warn(f'No data for {p_id} - {session_id}. Skipping...')
                continue
            if 'Annotations' in data.keys():
                warnings.warn(f'Annotations found in {p_id} - {session_id}. Check file')
                # left_foot = get_apdm_sensor_data_by_location_hardcoded(data, 'left Foot')
                data_new = dict()
                for new_loc, old_loc in zip(SENSOR_LOCATIONS, SENSOR_LOCATIONS_APDM.values()):
                    data_loc = get_apdm_sensor_data_by_location_hardcoded(data, old_loc)
                    data_new[new_loc] = {
                        'acc': data_loc['Accelerometer'],
                        'gyr': data_loc['Gyroscope'],
                        'timestamp': data_loc['Time']
                    }
                save_dict_to_hdf5(data_new, filepath)
            foo = 1
