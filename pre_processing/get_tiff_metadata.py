from os.path import sep
import os
import pickle as pkl
from natsort import natsorted
from tqdm import tqdm
import exifread
import numpy as np

def get_tiff_metadata(data_path, metadata_file, overwrite = False):

    # Load metadata
    with open('{0}{1}{2}'.format(data_path, sep, metadata_file), 'rb') as f:
        metadata = pkl.load(f)
    frame_times_file = metadata['frame_times_file']
    with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'rb') as f:
        output = pkl.load(f)
    # Check if tiff file metadata is already stored
    try:
        tiff_metadata = output['tiff_metadata']
        n_tiff_files = tiff_metadata['n_tiff_files']
        raw_tiff_files = tiff_metadata['raw_tiff_files']
        tiff_time_from_start_sec = tiff_metadata['tiff_time_from_start_sec']
        bad_timestamps = tiff_metadata['bad_timestamps']
        print('Tiff metadata loaded')
    except:
        overwrite = True
        print('Could not load tiff metadata from {0}. Overwriting'.format(frame_times_file))

    if overwrite:
        raw_tiff_image_path = metadata['raw_tiff_image_path']
        sessions_to_process = metadata['sessions_to_process']

        tiff_time_from_start_sec = {}
        bad_timestamps = {}
        n_tiff_files = {session: 0 for session in sessions_to_process}
        raw_tiff_files = {}

        for session in sessions_to_process:

            print('Session {0}: counting raw tiff files and getting timestamps'.format(session))
            # Order tiff files in ascending order
            raw_tiff_files[session] = os.listdir(raw_tiff_image_path[session])
            raw_tiff_files[session] = [file for file in raw_tiff_files[session] if file.endswith('.tif')]
            n_tiff_files[session] = len(raw_tiff_files[session])
            raw_tiff_files_ordered = natsorted(raw_tiff_files[session])

            tiff_time_from_start_sec[session] = np.zeros(n_tiff_files[session])
            bad_timestamps[session] = np.zeros(n_tiff_files[session])

            for file_no in tqdm(range(n_tiff_files[session])):

                file = raw_tiff_files_ordered[file_no]
                file_path = '{0}{1}{2}'.format(raw_tiff_image_path[session], sep, file)
                with open(file_path, 'rb') as f:
                    tags = exifread.process_file(f)
                f.close()
                tag = tags['Image ImageDescription']
                values = tag.values

                loc = values.find('Time_From_Last')
                #hrs = int(values[loc + 18:loc + 20])
                #minutes = int(values[loc + 21:loc + 23])
                try:
                    sec = float(values[loc + 24:loc + 30])
                except:
                    sec = 0.0024
                    bad_timestamps[session][file_no] = 1
                if file_no == 0:
                    tiff_time_from_start_sec[session][file_no] = sec
                else:
                    tiff_time_from_start_sec[session][file_no] = sec + tiff_time_from_start_sec[session][file_no - 1]

        tiff_metadata = {'n_tiff_files': n_tiff_files,
                         'raw_tiff_files': raw_tiff_files,
                         'tiff_time_from_start_sec': tiff_time_from_start_sec,
                         'bad_timestamps': bad_timestamps
                         }
        output['tiff_metadata'] = tiff_metadata
        with open('{0}{1}{2}'.format(data_path, sep, frame_times_file), 'wb') as f:
            pkl.dump(output, f)

    return tiff_metadata
