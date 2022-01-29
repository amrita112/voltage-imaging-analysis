
import numpy as np
from pre_processing import get_jitter

def check_tiff_timestamps(n_tiff_files, n_daq_frames, bad_timestamps, tiff_time_from_start_sec, frame_times, daq_frame_samples, n_trials):

    if n_tiff_files == n_daq_frames:
        print('Number of tiff files is same as number of camera triggers. Using camera triggers as frame times.')
        tiff_file_numbers = np.array(list(range(n_tiff_files))).astype(int)
    else:
        if n_daq_frames == 0:
            print('No camera output triggers')
            daq_frame_samples = []
            tiff_file_numbers = []
        else:
            print('{0} tiff files and {1} camera triggers'.format(n_tiff_files, n_daq_frames))
            assert(len(bad_timestamps) == len(tiff_time_from_start_sec))

            # Calculate inter frame interval based on tiff metadata
            tiff_ifi = np.diff(tiff_time_from_start_sec)
            max_tiff_ifi = np.max(tiff_ifi)
            med_ifi = np.min(tiff_ifi) + 0.1*(max_tiff_ifi - np.min(tiff_ifi))
            n_large_ifis = np.sum(tiff_ifi > med_ifi)

            # Calculate inter frame interval based on daq frame samples
            daq_ifi = np.diff(frame_times)
            max_daq_ifi = np.max(daq_ifi)

            if np.logical_or(np.sum(bad_timestamps) == 0, n_large_ifis == n_trials):
                print('All tiff timestamps are readable OR number of large IFIs from Tiff timestamps is equal to expected number of trials')

                if max_tiff_ifi <= max_daq_ifi*1.1:
                    output = get_jitter.get_jitter(tiff_time_from_start_sec, frame_times)
                    daq_frame_samples = daq_frame_samples[output['first_frame_trigger']:output['last_frame_trigger']]
                    tiff_file_numbers = list(range(output['first_tiff_file'], output['last_tiff_file']))

                else:
                    good_ifis = tiff_ifi <= max_daq_ifi*1.1
                    if np.sum(good_ifis) < 0.8*n_tiff_files:
                        print('Less than 80% of IFIs from tiff timestamps are below the expected IFI from daq data')
                        daq_frame_samples = []
                        tiff_file_numbers = []
                    else:
                        print('Using longest sequence of IFIs < largest expected from daq data')
                        (first_tiff_file, last_tiff_file) = longest_sequence(good_ifis)

                        output = get_jitter.get_jitter(tiff_time_from_start_sec[first_tiff_file:last_tiff_file], frame_times)
                        daq_frame_samples = daq_frame_samples[output['first_frame_trigger']:output['last_frame_trigger']]
                        tiff_file_numbers = list(range(first_tiff_file + output['first_tiff_file'], first_tiff_file + output['last_tiff_file']))

            else:
                print('Not all timestamps are readable, and number of large IFIs is not equal to expected number of trials')
                if np.sum(bad_timestamps) > 0.8*n_tiff_files:
                    print('More than 80% of tiff timestamps are corrupted')
                    daq_frame_samples = []
                    tiff_file_numbers = []
                else:
                    print('Using longest sequence of non-corrupted IFIs')
                    (first_tiff_file_non_corrupt, last_tiff_file_non_corrupt) = longest_sequence(1 - bad_timestamps)
                    tiff_ifi = np.diff(tiff_time_from_start_sec[first_tiff_file_non_corrupt:last_tiff_file_non_corrupt])
                    max_tiff_ifi = np.max(tiff_ifi)

                    if max_tiff_ifi <= max_daq_ifi*1.1:
                        print('In tiffs with non-corrupted timestamps, all IFIs are within expected limits')
                        output = get_jitter.get_jitter(tiff_time_from_start_sec[first_tiff_file_non_corrupt:last_tiff_file_non_corrupt], frame_times)
                        daq_frame_samples = daq_frame_samples[output['first_frame_trigger']:output['last_frame_trigger']]
                        tiff_file_numbers = list(range(first_tiff_file_non_corrupt + output['first_tiff_file'], first_tiff_file_non_corrupt + output['last_tiff_file']))

                    else:
                        print('In tiffs with non-corrupted timestamps, not all IFIs are within expected limits')
                        good_ifis = tiff_ifi <= max_daq_ifi*1.1
                        if np.sum(good_ifis) < 0.2*n_tiff_files:
                            print('Less than 20% of IFIs from tiff timestamps are below the expected IFI from daq data')
                            daq_frame_samples = []
                            tiff_file_numbers = []
                        else:
                            print('Using longest sequence of IFIs < largest expected from daq data')
                            (first_tiff_file_good_ifi, last_tiff_file_good_ifi) = longest_sequence(good_ifis)

                            output = get_jitter.get_jitter(tiff_time_from_start_sec[first_tiff_file_good_ifi + first_tiff_file_non_corrupt:last_tiff_file_good_ifi + first_tiff_file_non_corrupt],
                                                            frame_times)
                            daq_frame_samples = daq_frame_samples[output['first_frame_trigger']:output['last_frame_trigger']]
                            tiff_file_numbers = list(range(first_tiff_file_non_corrupt + first_tiff_file_good_ifi + output['first_tiff_file'],
                                                           first_tiff_file_non_corrupt + first_tiff_file_good_ifi + output['last_tiff_file']))


    assert(len(daq_frame_samples) == len(tiff_file_numbers))

    return (daq_frame_samples, tiff_file_numbers)

def longest_sequence(a):
    # Find the longest consequtive sequence of ones in a binary array a
    starts = []
    stops = []
    start = np.where(a)[0][0]
    flag = 1
    for stop in range(start + 1, len(a)):
        if flag:
            if a[stop] == 0:
                starts = np.append(starts, start)
                stops = np.append(stops, stop)
                flag = 0
        else:
            # flag = 0
            if a[stop]:
                start = stop
                flag = 1
    if flag:
        starts = np.append(starts, start)
        stops = np.append(stops, stop)

    lengths = stops - starts
    max_start = starts[np.argmax(lengths)]
    max_stop = stops[np.argmax(lengths)]

    return (int(max_start), int(max_stop))
