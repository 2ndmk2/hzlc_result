import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import interpolate
import scipy.integrate as integrate
import scipy.special as special
import pandas as pd



def tomoe_x_conv(x):
    return 400 + (x - 567) *500/ (995-567)

def tomoe_y_conv(x):
    return -(x - 689)/(689-203)

def make_tomoe_sensitivity(lambda_min=200, lambda_max=5200, n_sample = 5000):
    x_arr = np.array([541, 567, 609, 651, 695, 738, 781, 824, 866, 909, 952, 996, 1039, 1080, 1124 ])
    y_arr = np.array([552, 470, 370, 360, 373, 412, 465, 518, 562, 600, 630, 654, 672, 682, 687])
    lambda_raw= tomoe_x_conv(x_arr)
    eff_raw = tomoe_y_conv(y_arr)
    
    f = interpolate.interp1d(lambda_raw,eff_raw, fill_value = "extrapolate")
    lambda_sample = np.linspace(lambda_min, lambda_max,n_sample)
    eff_sample = f(lambda_sample)
    eff_sample[eff_sample <0] = 0
    func_sensitivty = interpolate.interp1d(lambda_sample,eff_sample , fill_value = "extrapolate")
    
    return func_sensitivty, lambda_sample, eff_sample, lambda_raw, eff_raw
def make_tess_sensitivty(file="/Users/masatakaaizawa/research/heso/tomoe_analysis/modules/tess_response/tess-response-function-v2.0.csv"):
    df_tess = pd.read_csv(file)
    lambda_arr = df_tess["Lambda"]
    trans_arr = df_tess["trans"]
    f_sample, lambda_sample, eff_sample = interpolate_sensitivity(lambda_arr, trans_arr )
    return f_sample

def make_ultracam():
    lambda_arr = np.linspace(200, 1000, 10000)
    trans_arr = np.zeros(len(lambda_arr))
    trans_arr[np.abs((lambda_arr-601))<20] = 1
    f_sample, lambda_sample, eff_sample = interpolate_sensitivity(lambda_arr, trans_arr)

    return f_sample

def make_galex():
    lambda_arr = np.linspace(200, 1000, 10000)
    trans_arr = np.zeros(len(lambda_arr))
    trans_arr[np.abs((lambda_arr-601))<20] = 1
    f_sample, lambda_sample, eff_sample = interpolate_sensitivity(lambda_arr, trans_arr)

    return f_sample

def interpolate_sensitivity(x, y, x_sample_min=200, x_sample_max=5200, n_sample = 5000):
    f = interpolate.interp1d(x,y, fill_value = "extrapolate")
    lambda_sample = np.linspace(x_sample_min, x_sample_max, 5000)
    eff_sample = f(lambda_sample)
    eff_sample[eff_sample <0] = 0
    f_sample = interpolate.interp1d(lambda_sample,eff_sample , fill_value = "extrapolate")
    return f_sample, lambda_sample, eff_sample


def plankfunc_return_func(lambda_now, T):
    """
    lambda_arr: wavelength (nm)
    """
    lambda_cm = lambda_now * 10**(-7)
    k = 1000 * (100 * 100) * 1.38064852* 10**(-23)# (cm)^2 g s-2 K-1
    h = 1000 * (100 * 100) * 6.62607004*10**(-34)# (cm)2 g / s
    c = 100* 299792458 # cm/s
    angular_factor = np.pi
    B_lambda_nm = 10 **(-7) *angular_factor * (2 * h * c*c/lambda_cm **5)/(np.exp(h * c/(lambda_cm  * k * T)) - 1) ##erg/s/nm
    return B_lambda_nm

def plank_func_sensivitiy_integrate(lambda_min, lambda_max, T,  sensitivity = None):
    
    k = 1000 * (100 * 100) * 1.38064852* 10**(-23)# (cm)^2 g s-2 K-1
    h = 1000 * (100 * 100) * 6.62607004*10**(-34)# (cm)2 g/s
    c = 100* 299792458 # cm/s
    sigma_SB = (2 * (np.pi **5) * k**4)/(15 * (c**2) * h**3) ## 

    if sensitivity is None:
        int_result = sigma_SB * T**4 
        return int_result
    else:
        int_result = integrate.quad(lambda x: plankfunc_return_func(x, T) * sensitivity(x), lambda_min,lambda_max, limit=1000)
        return int_result[0]
    
def plankfunc_return_func(lambda_now, T):
    
    lambda_cm = lambda_now * 10**(-7)
    k = 1000 * (100 * 100) * 1.38064852* 10**(-23)# (cm)^2 g s-2 K-1
    h = 1000 * (100 * 100) * 6.62607004*10**(-34)# (cm)2 g / s
    c = 100* 299792458 #cm /s
    angular_factor = np.pi 
    B_lambda_nm = 10 **(-7) *angular_factor * (2 * h * c*c/lambda_cm **5)/(np.exp(h * c/(lambda_cm  * k * T)) - 1) ##erg/s/nm

    return B_lambda_nm


def plankfunc(T, lambda_arr):
    
    lambda_arr_cm = lambda_arr * 10**(-7)
    k = 1000 * (100 * 100) * 1.38064852* 10**(-23)# (cm)^2 g s-2 K-1
    h = 1000 * (100 * 100) * 6.62607004*10**(-34)# (cm)2 g / s
    c = 100* 299792458 #cm /s
    angular_factor = np.pi 
    B_lambda_cm = angular_factor * (2 * h * c*c/lambda_arr_cm**5)/(np.exp(h * c/(lambda_arr_cm * k * T)) - 1) ##erg/s/cm
    B_lambda_nm = 10 **(-7) * B_lambda_cm ##erg/s/nm
    return B_lambda_nm

def compute_flare_star_energy(lambda_min, lambda_max,  T_eff_star,  T_eff_flare):
    
    f_sample, lambda_sample, eff_sample, lambda_raw, eff_raw = make_tomoe_sensitivity()    
    tot_Energy_star = plank_func_sensivitiy_integrate(10, 2 * 100000, T_eff_star, None)
    tot_Energy_flare = plank_func_sensivitiy_integrate(10, 2 * 100000, T_eff_flare, None)
    Energy_flare = plank_func_sensivitiy_integrate(lambda_min, lambda_max, T_eff_flare, f_sample)
    Energy_star = plank_func_sensivitiy_integrate(lambda_min, lambda_max, T_eff_star, f_sample)
    return tot_Energy_star,tot_Energy_flare,Energy_flare,Energy_star


def compare_tomoe_ultracam_tess(T_eff_star,  T_eff_flare, lambda_min=200, lambda_max=1200, file_tess = "/Users/masatakaaizawa/research/heso/tomoe_analysis/modules/tess_response/tess-response-function-v2.0.csv"):

    f_sample_tomoe, lambda_sample, eff_sample, lambda_raw, eff_raw = make_tomoe_sensitivity()    
    f_sample_tess= make_tess_sensitivty(file_tess)    
    #f_sample_ultracam= make_ultracam()    
    Energy_flare_tomoe = plank_func_sensivitiy_integrate(lambda_min, lambda_max, T_eff_flare, f_sample_tomoe)
    Energy_star_tomoe = plank_func_sensivitiy_integrate(lambda_min, lambda_max, T_eff_star, f_sample_tomoe)
    Energy_flare_tess = plank_func_sensivitiy_integrate(lambda_min, lambda_max, 
        T_eff_flare, f_sample_tess)
    Energy_star_tess = plank_func_sensivitiy_integrate(lambda_min, lambda_max, T_eff_star, f_sample_tess)
    #Energy_flare_ultracam = plank_func_sensivitiy_integrate(lambda_min, lambda_max, T_eff_flare, f_sample_ultracam)
    #Energy_star_ultracam = plank_func_sensivitiy_integrate(lambda_min, lambda_max, T_eff_star, f_sample_ultracam)

    ratio_tomoe =Energy_flare_tomoe/Energy_star_tomoe
    ratio_tess =Energy_flare_tess /Energy_star_tess 
    ratio_ultracam =plankfunc_return_func(601, T_eff_flare)/plankfunc_return_func(601, T_eff_star)
    return ratio_tomoe , ratio_tess, ratio_ultracam




