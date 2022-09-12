import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import interpolate
import scipy.integrate as integrate
import scipy.special as special

def residual_1comp(params, x, data, eps_data):
    t_rise= params['t_rise']
    t_peak = params["t_peak"]
    f_peak = params["f_peak"]
    t_1_2_decay= params["t_1_2_decay"]
    flux_model = model_flare_1comp(x, t_rise, t_peak, f_peak, t_1_2_decay)
    return  (data - flux_model)/eps_data

    
def model_fit_1comp(time, flux, flux_err,  t_rise_init,  t_peak_init, f_peak_init, t_1_2_decay_init):
    params = Parameters()

    params.add('t_rise', t_rise_init , min = 2)
    params.add('t_peak', t_peak_init)
    params.add('f_peak', value=f_peak_init)
    params.add('t_1_2_decay', value=t_1_2_decay_init)    
    
    result = minimize(residual_1comp, params, args=(time, flux, flux_err ))
    out = result.params
    model_flux = model_flare_1comp(time,out["t_rise"],  out["t_peak"], out["f_peak"], out["t_1_2_decay"])
    
    ## parameter retrieval
    para_arr= []
    err_arr = []
    for name, param in result.params.items():
        para_arr.append(param.value)
        err_arr.append(param.stderr)
    para_arr = np.array(para_arr)
    err_arr = np.array(err_arr)
    return model_flux, para_arr, err_arr

def model_flare_comp2(time, t_rise,  t_peak, f_peak, t_1_2_decay,  t_1_2_decay_2, fraction):
    t_start = t_peak - t_rise
    flux_model = np.zeros(len(time))
    time_linear = time - t_start    
    flux_model[time< t_start] = 0
    mask = (time>t_start) * (time< t_peak)
    flux_model[ mask] = f_peak * time_linear[mask]/ (t_rise)
    mask =  (time>t_peak)
    time_dif = time - t_peak
    sigma = t_1_2_decay/np.log(2)
    sigma_2 = t_1_2_decay_2/np.log(2)

    flux_model[mask]  = f_peak *  ( fraction *  np.exp( -  time_dif[mask]/ sigma)   + (1- fraction) * np.exp( -  time_dif[mask]/ sigma_2) )
    return flux_model

def residual_2comp(params, x, data, eps_data):
    t_rise= params['t_rise']
    t_peak = params["t_peak"]
    f_peak = params["f_peak"]
    t_1_2_decay= params["t_1_2_decay"]
    t_1_2_decay_2= params["t_1_2_decay_2"]
    fraction= params["fraction"]
    flux_model = model_flare_comp2(x, t_rise, t_peak, f_peak, t_1_2_decay , t_1_2_decay_2, fraction  )
    return  (data - flux_model)/eps_data


def model_fit_2comp(time, flux, flux_err,  t_rise_init,  t_peak_init, f_peak_init, t_1_2_decay_init,  t_1_2_decay_2_init, fraction_init ):
    params = Parameters()
    params.add('t_rise', t_rise_init , min = 2)
    params.add('t_peak', t_peak_init)
    params.add('f_peak', value=f_peak_init)
    params.add('t_1_2_decay', value=t_1_2_decay_init)    
    params.add('ratio_t_1_2', value=t_1_2_decay_2_init/t_1_2_decay_init, min = 1.5)    
    params.add('t_1_2_decay_2', expr="t_1_2_decay * ratio_t_1_2")
    params.add("fraction", value= fraction_init, min = 0.5, max = 1)
    result = minimize(residual_2comp, params, args=(time, flux, flux_err ))
    out = result.params
    model_flux = model_flare_comp2(time,out["t_rise"],  out["t_peak"], out["f_peak"], out["t_1_2_decay"],  out["t_1_2_decay_2"] ,  out["fraction"])
    ## parameter retrieval
    para_arr= []
    err_arr = []
    for name, param in result.params.items():
        para_arr.append(param.value)
        err_arr.append(param.stderr)
    para_arr = np.array(para_arr)
    err_arr = np.array(err_arr)
    return model_flux, para_arr, err_arr