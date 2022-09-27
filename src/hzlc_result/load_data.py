import pandas as pd
import numpy as np
import astropy.units as u
import pandas as pd

def load_flare_csv(file_name):

    df_flare_can = pd.read_csv(file_name)
    dates = np.array(df_flare_can['date'])
    gaia_ids = np.array(df_flare_can['Gaia ID'],dtype = int)
    tic_ids = np.array(df_flare_can['TIC ID'])
    t_half = np.array(df_flare_can['t_1_2 (s)'])
    t_rise = np.array(df_flare_can['rise (s)'])
    t_peak = np.array(df_flare_can['tpeak'])
    mass = np.array(df_flare_can['mass'])
    rad = np.array(df_flare_can['rad'])
    teff = np.array(df_flare_can['Teff'])
    Leff = np.array(df_flare_can['lum'])
    tplotmin = np.array(df_flare_can['tplotmin'],dtype = int)
    tplotMax = np.array(df_flare_can['tplotMax'], dtype = int)
    plot_flux_type = np.array(df_flare_can['plot_flux_type'], dtype = str)
    return dates, gaia_ids, tic_ids, t_half, t_rise, t_peak, mass, rad, teff, Leff, tplotmin, tplotMax, plot_flux_type

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


def tess_gunther_load(file_name = "/Users/masatakaaizawa/research/heso/tomoe_analysis/table_for_tess/gunther_2020.txt"):
    file = open(file_name, "r")
    lines = file.readlines()
    t_fwhm_arr = []
    teff_arr = []
    Ebol_arr = []
    t_fwhm_low_arr = []
    t_fwhm_upp_arr = []
    for line in lines:
        try:
            t_fwhm = float(line[75:82])
            t_fwhm_lower = float(line[83:90])
            t_fwhm_upper = float(line[91:98])
            teff = float(line[168:173])
            Ebol= float(line[99:108])
            t_fwhm_arr.append(t_fwhm)
            t_fwhm_low_arr .append(t_fwhm_lower)
            t_fwhm_upp_arr .append(t_fwhm_upper)
            teff_arr.append(teff)
            Ebol_arr.append(Ebol)
            print(t_fwhm, )
        except:
            pass
    t_fwhm_arr  = 3600*24 * np.array( t_fwhm_arr)
    t_fwhm_low_arr  = 3600*24 * np.array( t_fwhm_low_arr)
    t_fwhm_upp_arr  = 3600*24 * np.array( t_fwhm_upp_arr)
    teff_arr  = np.array(teff_arr)
    Ebol_arr  = np.array(Ebol_arr)
    return t_fwhm_arr, t_fwhm_low_arr, t_fwhm_upp_arr, teff_arr, Ebol_arr

def load_lc_model_and_data(file):
    data = np.load(file, allow_pickle = True)

    try:

        time = data["time"]
        flux = data["flux"]
        flux_err = data["flux_err"]
        mean_muy = data["mean_model"]
        low_model = data["low_model"]
        upper_model = data["upper_model"]
        hpdi_muy = [low_model, upper_model]
        return time, flux, flux_err, mean_muy, hpdi_muy

    except:
        time = data["time"]
        flux = data["flux"]
        flux_err = data["flux_err"]
        mean_muy = data["mean_model"]
        low_model = data["low_model"]
        upper_model = data["upper_model"]
        hpdi_muy = [low_model, upper_model]
        return time, flux, flux_err, mean_muy, hpdi_muy


def load_lc_model_and_data_205(file):
    data = np.load(file, allow_pickle = True)


    time = data["time_mask"]
    flux = data["flux_mask"]
    flux_err = data["flux_err_mask"]
    mean_muy = data["mean_model"]
    low_model = data["low_model"]
    upper_model = data["upper_model"]
    hpdi_muy = [low_model, upper_model]
    return time, flux, flux_err, mean_muy, hpdi_muy


def load_csv_for_params(file_name):
    df = pd.read_csv(file_name)
    tic_ids = df["TIC ID"].values
    para_names_for_csv = ["t_rise[sec]_50","t_peak[sec]_50", "flare count", "f_peak_50", "t_peak_duration_50", "t_1_2_decay[sec]_50", 't_ratio_50', 't_1_2_decay_2[sec]_50', "fraction_50"]
    t_rise = df["t_rise[sec]_50"].values
    t_peak = df["t_peak[sec]_50"].values
    f_peak = df["f_peak_50"].values
    t_1_2_decay =df["t_1_2_decay[sec]_50"].values
    t_ratio = df["t_ratio_50"].values
    t_1_2_decay_2 = df["t_1_2_decay_2[sec]_50"] 
    fraction = df["fraction_50"].values
    t_peak_duration= df["t_peak_duration_50"].values
    flare_count= df["flare count"].values
    ED= df["ED"].values
    return tic_ids, t_rise, t_peak, f_peak, t_1_2_decay, t_ratio, t_1_2_decay_2, fraction, t_peak_duration, flare_count, ED

def get_complete_source_ids(df_catalog_target):
    ''' Obtain the complete source IDs from df_catalog

    Args:
        df_catalog_all_stars : 
    
    Returns:
        source_ids_all : list of complete soruce IDs
    
    '''
    source_ids = []
    list_of_sourceid = np.array(df_catalog_target['source_id'])
    list_of_catalogid = np.array(df_catalog_target['catalog_name'])
    
    for i in range(len(list_of_catalogid)):
        source_ids.append(str(list_of_catalogid[i])+ '_' + str(list_of_sourceid[i]))

    return source_ids
