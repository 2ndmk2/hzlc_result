import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import arviz
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

def poly_func_all(x,a, b, c, d):
    return jnp.polyval(jnp.array([d, c, b, a, 0]), x)#a * x + b * x**2 + c* x**3 + d * x**4

def width_func(x, xstart, xend):
    return jnp.heaviside(x- xstart,0) * ( 1- jnp.heaviside(x-xend, 0))

def width_func_np(x, xstart, xend):
    return np.heaviside(x- xstart,0) * ( 1- np.heaviside(x-xend, 0))


def make_dirs(para_names, samples):
    
    para_dic = {}
    for para_name in para_names:
        para_dic[para_name] = samples[para_name]
    return para_dic



def make_dir_for_init(para_names, para_arr):
    
    para_dic = {}
    for (i, para_name) in enumerate(para_names):
        para_dic[para_name] = para_arr[i]
    return para_dic

def make_median_dirs(para_names, samples):
    
    para_dic = {}
    for para_name in para_names:
        para_dic[para_name] = np.median(samples[para_name])
    return para_dic

def make_from_index(para_names, samples):
    
    para_dic = {}
    for para_name in para_names:
        para_dic[para_name] = np.median(samples[para_name])
    return para_dic



def init_for_poly_const(dic_non_poly):
    dic_non_poly["a"] = dic_non_poly["f_peak"]/dic_non_poly["t_rise"]
    dic_non_poly["t_peak_duration"] = 3
    return dic_non_poly



def init_for_poly(dic_non_poly):
    dic_non_poly["a"] = dic_non_poly["f_peak"]/dic_non_poly["t_rise"]
    dic_non_poly["b"] =  0.0
    dic_non_poly["c"] =  0.0
    return dic_non_poly

def calc_percentile(mcmc_samples, para_names):
    para_arr = []

    for name in para_names:
        values = np.percentile(mcmc_samples[name], [14, 50, 86])
        para_arr.append(values)

    return para_arr

import scipy.integrate as integrate

def calc_ED_for_model(samples, para_name, tplotMax=1000,i = 0, second_flare = False):

    para_dic = make_from_index( para_name, samples)
    x_max =tplotMax
    if second_flare:
        x_min = -para_dic["t_rise_two"]
    else:
        x_min = -para_dic["t_rise"]
    print(x_min, x_max)
    x_arr = np.linspace(x_min, x_max, 10000)
    if second_flare:
        ED = integrate.quadrature(lambda x: model_flare_4th_for_int_two(x, para_dic), x_min ,x_max)
        plt.plot(x_arr, model_flare_4th_for_int_two(x_arr, para_dic))
    else:
        ED = integrate.quadrature(lambda x: model_flare_4th_for_int(x, para_dic), x_min ,x_max)
        plt.plot(x_arr, model_flare_4th_for_int(x_arr, para_dic))
    plt.show()
    return ED[0]

def fast_calc_ed(samples, para_name):
    mcmc_result = make_median_dirs(para_name, samples)

    t_rise = mcmc_result["t_rise"]
    f_peak = mcmc_result["f_peak"]
    t_1_2_decay = mcmc_result["t_1_2_decay"]
    t_1_2_decay_2 = mcmc_result["t_1_2_decay_2"]
    t_peak_duration = mcmc_result["t_peak_duration"]
    fraction = mcmc_result["fraction"]

    ED_rise = 0.5 * t_rise * f_peak
    ED_fast = fraction * t_1_2_decay * f_peak
    ED_const = f_peak * t_peak_duration
    ED_slow = (1-fraction) * t_1_2_decay_2 * f_peak
    ED_sum = ED_rise + ED_fast + ED_slow + ED_const

    return ED_rise, ED_const , ED_fast, ED_slow, ED_sum 


def model_flare_4th_for_int(x, mcmc_result):

    t_rise = mcmc_result["t_rise"]
    t_peak_duration = mcmc_result["t_peak_duration"]
    a = mcmc_result["a"]
    f_peak = mcmc_result["f_peak"]
    t_1_2_decay = mcmc_result["t_1_2_decay"]
    t_1_2_decay_2 = mcmc_result["t_1_2_decay_2"]
    fraction = mcmc_result["fraction"]
 
    num_poly = 4
    minus_one = (-1) ** np.arange(num_poly-1) 
    d =  np.sum(a * minus_one)-1
    t_start = - t_rise   
    poly_value = np.ones(num_poly+1)
    poly_value = np.append(np.array([d]), np.flip(a))  
    poly_value_flux = np.append(poly_value, np.array([1]))
    time_linear = x - t_start 
    time_linear_for_rise = (x)/t_rise
    time_dif = x - (t_peak_duration)    


    poly_model = f_peak * np.polyval( poly_value_flux, time_linear_for_rise )
    model_rise  = width_func_np(time_linear, 0, t_rise) * poly_model
    model_peak_const  =  f_peak * width_func_np(time_linear, t_rise, t_rise + t_peak_duration) 
    mu = model_rise +model_peak_const +  f_peak * np.heaviside(time_dif,0) *  \
    ( fraction *  np.exp( -  time_dif/ t_1_2_decay)   + (1- fraction) * np.exp( -  time_dif/ t_1_2_decay_2) )

    return mu


def model_flare_4th_for_int_two(x, mcmc_result):

    t_rise = mcmc_result["t_rise_two"]
    t_peak_duration = mcmc_result["t_peak_duration_two"]
    a = mcmc_result["a_two"]
    f_peak = mcmc_result["f_peak_two"]
    t_1_2_decay = mcmc_result["t_1_2_decay_two"]
    t_1_2_decay_2 = mcmc_result["t_1_2_decay_2_two"]
    fraction = mcmc_result["fraction_two"]
    num_poly = 4
    minus_one = (-1) ** np.arange(num_poly-1) 
    d =  np.sum(a * minus_one)-1
    t_start = - t_rise   
    poly_value = np.ones(num_poly+1)
    poly_value = np.append(np.array([d]), np.flip(a))  
    poly_value_flux = np.append(poly_value, np.array([1]))
    time_linear = x - t_start 
    time_linear_for_rise = (x)/t_rise
    time_dif = x - (t_peak_duration)    

    poly_model = f_peak * np.polyval( poly_value_flux, time_linear_for_rise )
    model_rise  = width_func_np(time_linear, 0, t_rise) * poly_model
    model_peak_const  =  f_peak * width_func_np(time_linear, t_rise, t_rise + t_peak_duration) 
    mu = model_rise +model_peak_const +  f_peak * np.heaviside(time_dif,0) *  \
    ( fraction *  np.exp( -  time_dif/ t_1_2_decay)   + (1- fraction) * np.exp( -  time_dif/ t_1_2_decay_2) )

    return mu

def compute_ED(time, flux):
    y = interpolate.interp1d(time, flux, kind="quadratic")
    time_re = np.linspace(np.min(time), np.max(time), 10000)
    ED = integrate.quadrature(y, np.min(time), np.max(time))
    return ED

def model_flare_4th_from_dic(time, flux, flux_err, para_dic):
    
    ### parameter
    t_peak_start = para_dic["t_peak"]
    t_rise = para_dic["t_rise"]
    t_peak_duration = para_dic['t_peak_duration']
    f_peak = para_dic['f_peak']
    t_1_2_decay = para_dic['t_1_2_decay']
    t_ratio = para_dic['t_ratio']
    t_1_2_decay_2 =  t_ratio * t_1_2_decay
    fraction = para_dic['fraction']
    t_start = t_peak_start - t_rise   
    a = para_dic['a']
    b = para_dic['b']
    c =para_dic['c']
    d =  (a - b + c -1 )

    time_linear = time - t_start 
    time_linear_for_rise = (time - t_peak_start)/t_rise
    time_dif = time - (t_peak_start + t_peak_duration)
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2

    poly_model = f_peak * np.polyval( np.array([d, c, b, a, 1]), time_linear_for_rise )
    model_rise  = width_func_np(time_linear, 0, t_rise) * poly_model# * jnp.exp((time -t_peak_start)/ t_tau_rise)
    model_peak_const  = width_func_np(time_linear, t_rise, t_rise + t_peak_duration) * f_peak
    mu = model_rise +model_peak_const +  np.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  np.exp( -  time_dif/ sigma)   + (1- fraction) * np.exp( -  time_dif/ sigma_2) )
    loglikelihood = np.sum(- 0.5 * (flux- mu)**2/flux_err**2 - np.log(flux_err))
    return mu


def model_flare_4th(time, flux, flux_err, num_poly):
    
    ### parameter
    t_peak_start = numpyro.sample('t_peak',  dist.Uniform(-100, 100))
    t_rise = numpyro.sample('t_rise',  dist.Uniform(4, 300))
    t_peak_duration = numpyro.sample('t_peak_duration', dist.Uniform(0, 300))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 100))
    t_1_2_decay = numpyro.sample('t_1_2_decay',dist.Uniform(.1, 1000))
    t_ratio = numpyro.sample('t_ratio',  dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.99))
    t_start = t_peak_start - t_rise   
    lows = np.ones(num_poly-1)*(-10)
    highs = np.ones(num_poly-1) * 10
    lows[0] = 0
    a = numpyro.sample('a', dist.Uniform(lows, highs))  
    minus_one = (-1) ** np.arange(num_poly-1)  
    d = numpyro.deterministic('d', np.sum(a * minus_one)-1)
    poly_value = np.ones(num_poly+1)
    poly_value = jnp.append(jnp.array([d]), jnp.flip(a))
    poly_value_flux = jnp.append(poly_value, jnp.array([1]))
    poly_value_deri = poly_value * np.flip(np.arange(num_poly)+1)

    time_linear = time - t_start 
    time_linear_for_rise = (time - t_peak_start)/t_rise
    time_dif = time - (t_peak_start + t_peak_duration)
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2

    poly_model_deri = width_func(time_linear, 0, t_rise) * jnp.polyval(poly_value_deri, time_linear_for_rise )
    poly_model = f_peak * jnp.polyval( poly_value_flux, time_linear_for_rise )
    model_rise  = width_func(time_linear, 0, t_rise) * poly_model# * jnp.exp((time -t_peak_start)/ t_tau_rise)
    model_peak_const  = width_func(time_linear, t_rise, t_rise + t_peak_duration) * f_peak
    mu = model_rise +model_peak_const +  jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = jnp.sum(- 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(model_rise)+0.0000001, 0))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(poly_model_deri)+0.0000001, 0))
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)



def care_4th_mcmc(time, mcmc_samples, para_names, rng_key_, file_head = "", num_poly=4):
    para_dic = make_dirs(para_names,mcmc_samples)
    """
    arviz.plot_trace(mcmc, var_names=["t_rise","t_peak", "f_peak", "t_1_2_decay", 't_ratio', 't_1_2_decay_2', "fraction"])
    plt.savefig(file_head + "_hmc2.pdf")
    plt.close()
    arviz.plot_pair(arviz.from_numpyro(mcmc),var_names = para_names, kind='kde',
    divergences=False,marginals=True)
    plt.savefig(file_head + "_posterior.pdf")
    plt.close()
    """

    pred = Predictive(model_flare_4th, para_dic, return_sites=["mu"])
    null_arr = jnp.ones(len(time))
    predictions = pred(rng_key_,time=time,flux=null_arr, flux_err = null_arr, num_poly = num_poly)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    hpdi_muy = hpdi(predictions["mu"], 0.9)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    para_arr = calc_percentile(mcmc_samples, para_names)
    return mean_muy, hpdi_muy, para_arr

def make_para_dict_for_4th_order(para_dic_for_init, num_poly):
    """
    para_dic_for_init["a"] = 1#para_dic_for_init["t_rise"]
    para_dic_for_init["b"] = 0
    para_dic_for_init["c"] = 0
    """
    poly_value = np.zeros(num_poly-1)
    poly_value[0] = 1
    para_dic_for_init["a"] = poly_value
    para_dic_for_init["t_peak_duration"] = .2
    return para_dic_for_init

def run_hmc_flare_4th(time, flux, flux_err, para_names,  num_warmup, num_samples, file_head, value_dic = None, dense_mass = True, num_poly = 4):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    print(rng_key_)
    if value_dic is not None:  
        kernel = NUTS(model_flare_4th, dense_mass=dense_mass, init_strategy=init_to_value(values=value_dic))
    else:
        kernel = NUTS(model_flare_4th, dense_mass=dense_mass)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, time=time, flux=flux, flux_err =flux_err, num_poly = num_poly)
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mean_muy, hpdi_muy, para_arr = care_4th_mcmc(time, mcmc.get_samples(), para_names, rng_key_, file_head ,num_poly = num_poly)
    return mean_muy, hpdi_muy, para_arr, mcmc_samples


def load_lc_data(file):
    data = np.load(file, allow_pickle = True)
    time = data["time"]
    flux = data["flux"]
    flux_err = data["flux_err"]
    mean_muy = data["mean_model"]
    low_model = data["low_model"]
    upper_model = data["upper_model"]
    hpdi_muy = [low_model, upper_model]
    return time, flux, flux_err, mean_muy, hpdi_muy
