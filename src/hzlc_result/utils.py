"""
Misc utilities

Author: Kojiro Kawana
"""
import os
import glob
import numpy as np
import datetime


def is_array_like(x):
    """
    Judge whether the argument type is array-like (list, numpy.array, pandas.Series).
    """
    try:
        list(x)
        return True
    except:
        return False


def get_slice_of_dictionay(dictionary, keys):
    """
    Get slice of Dictionary with the given keys.

    arguments
    =========
    dictionary : dict
    keys : list of str
        Keys you want to slice from dicitonary

    returns
    =======
    dict_slice : dict
        dicitonary with keys are overlaps between dictionary.keys() and keys
    """

    dict_slice = {}
    for key, values in dictionary.items():
        if key in keys:
            dict_slice[key] = values
    return dict_slice


def convert_time_from_datetime_to_str(arr_time, N_string_time):
    """
    Convert time in datetime.datetime object to string for output.

    arguments
    =========
    arr_time     : list/array of datetime
        Array of time in datetime.datetime object
    N_string_time: int
        Length of string to express time. Now we use 26 "YYYY-mm-dd HH:MM:SS.µµµµµµ" in UTC.
        This does not express time zone.

    return
    ======
    arr_time_string: numpy.array
        numpy.array(arr_time ,dtype="U{:d}".format(N_string_time))
    """
    arr_time_ = np.array(arr_time)
    func = np.vectorize(lambda x: np.unicode(x)[:N_string_time])
    arr_time_string = func(arr_time)

    return arr_time_string

def convert_time_from_str_to_datetime(time_str, timezone="utc"):
    """
    Convert datetime string to datetime.datetime object.

    arguments
    =========
    time_str : string of datetime
        Format must be ``%Y-%m-%dT%H:%M:%S.%f``
    timezone : None or "utc"

    return
    ======
    datetime : datetime.datetime

    Raise
    =====
    KeyError : Exception
        If timezone argument is not correct.
    """
    if timezone == "utc":
        return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc)
    elif timezone == None:
        return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        raise KeyError("timezone {:} is not suppored yet! Only None or ""utc"" is suppored now".format(timezone))

def extract_overlap_between_2_dataframes(df1, df2, subset, **kwargs):
    """
    Extract overlap between 2 DataFrames.

    arguments
    =========
    df1 : pandas.DataFrame
    df2 : pandas.DataFrame
    subset : column label or sequence of labels
        Only consider certain columns for identifying duplicates, by
        default use all of the columns.
    **kwargs : dict
        keyword arguments passed to df.duplicated()

    return
    ======
    indexes_overlapped: pandas.Index
        Indexes of  overlapped entries
    """
    df_total = df1[subset].append(df2[subset])
    mask = df_total.duplicated(**kwargs)
    indexes_overlapped = mask.index[mask]
    return indexes_overlapped

def concatenate_list(list_in):
    """
    Concatenate list.

    arguments
    =========
    list_in : list of list

    return
    ======
    list_out : list

    Examples
    ========
    >>> concatenate_list([[0], [1, 2], [[3, 4], 5]])
    [0, 1, 2, [3, 4], 5]

    """

    list_out = []
    for tmp in list_in:
        list_out.extend(tmp)
    return list_out

def replace_quarter_in_frame_id(frame_id:str):
    """
    Replace the location of quarter info.

    arguments
    =========
    frame_id : str
        `TMQ` must be removed

    return
    ======
    new_frame_id : str

    Examples
    ========
    >>> replace_quarter_in_frame_id('1201911070016636623')
    '2019110700166366123'
    """
    new_frame_id = frame_id[1:-2] + frame_id[0] + frame_id[-2:]
    return new_frame_id

def get_fits_frame_file_list(date, directory):
    """
    Get file lists in a directory with name ``*TMQ*``.

    arguments
    =========
    date : int or str
        Observed date
    directory : str
        Directory where files are searched

    return
    ======
    file_paths : list of str
        Sorted file paths
    """
    file_paths = glob.glob(os.path.join(directory, "*TMQ*{:}*".format(date)))
    file_paths.sort()
    return file_paths

def sort_frame_ids_by_time_order(frame_ids):
    """
    Sort FRAME_IDs by time order.
    The time is estimated by the frame_id. Sort preference is DATE-OBS => EXP_ID => DET_ID.


    arguments
    =========
    frame_ids: list_like
        FRAME_IDs. e.g. ["TMQ1202004010000000101", "TMQ1202004010000000102", ...]

    return
    ======
    frame_ids_sorted: numpy.array of str
        FRAME_IDs sorted by time order.
    sort_index: numpy.array of int
        indexes used to sort frame_ids. frame_ids[sort_index] == frame_ids_sorted.
    """

    frame_ids_renamed = np.array([replace_quarter_in_frame_id(frame_id.replace("TMQ", "")) for frame_id in frame_ids])
    sort_index = np.argsort(frame_ids_renamed)
    frame_ids_sorted = np.array(frame_ids)[sort_index]
    return frame_ids_sorted, sort_index
