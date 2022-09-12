import numpy as np


def galex_load(file_name="/Users/masatakaaizawa/research/heso/tomoe_analysis/table_for_galex/galex.txt"):
    file = open(file_name, "r")
    lines = file.readlines()
    dur_arr = []
    teff_arr = []
    Ebol_arr = []
    for line in lines:
        try:
            start = float(line[29:38])
            end = float(line[39:48])

            teff = float(line[79:84])
            Ebol= float(line[115:122])
            dur_arr.append(end-start)
            teff_arr.append(teff)
            Ebol_arr.append(Ebol)
        except:
            pass
    teff_arr  = np.array(teff_arr)
    dur_arr  = np.array(dur_arr)
    Ebol_arr  = np.array(Ebol_arr)
    return teff_arr, dur_arr, Ebol_arr
