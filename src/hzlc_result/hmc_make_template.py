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


def derive_mu_sigma(y_d, sigma_d):
    mu = numpyro.sample('mu', dist.Uniform(-1, 2))
    sigma = numpyro.sample('sigma', dist.Uniform(0.01, 1.0))
    sigma_dash = (sigma_d**2 + sigma**2)**0.5
    loglikelihood =  jnp.sum( - jnp.log(sigma_dash) - ((y_d - mu)**2)/(2*(sigma_dash)**2))
    numpyro.factor("loglike", loglikelihood)

def run_hmc_mu_sigma(y_d, sigma_d,value_dic, num_warmup = 1000, num_samples = 1000 ):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(derive_mu_sigma, init_strategy=init_to_value(values=value_dic))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, y_d=y_d, sigma_d=sigma_d)
    #mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mu = mcmc_samples["mu"]
    sigma= mcmc_samples["sigma"]

    return np.mean(mu), np.std(mu),np.mean(sigma), np.std(sigma)



def poly_func_all(x,a, b, c, d):
    return jnp.polyval(jnp.array([d, c, b, a, 0]), x)#a * x + b * x**2 + c* x**3 + d * x**4

def width_func(x, xstart, xend):
    return jnp.heaviside(x- xstart,0) * ( 1- jnp.heaviside(x-xend, 0))

def width_func_np(x, xstart, xend):
    return np.heaviside(x- xstart,0) * ( 1- np.heaviside(x-xend, 0))


def model_flare_comp2_after_jax(time, t_peak, t_rise, f_peak, t_1_2_decay,\
                                t_1_2_decay2, fraction):

    t_start = t_peak - t_rise    
    time_linear = time - t_start  
    time_dif = time - t_peak
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2
    
    mu = width_func(time_linear, 0, t_rise) * time_linear * f_peak/t_rise\
    + jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    return mu


def model_flare_comp2_jax(time, flux, flux_err):
    
    ### parameter
    t_peak = numpyro.sample('t_peak', dist.Uniform(-1, 1))
    t_rise = numpyro.sample('t_rise', dist.Uniform(.1, 2))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 2))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(.1, 100))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.9))

    t_start = t_peak - t_rise   
    time_linear = time - t_start  
    time_dif = time - t_peak
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2   
    mu = width_func(time_linear, 0, t_rise) * time_linear * f_peak/t_rise\
    + jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = - 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err)
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)


def model_flare_comp_all_poly_jax(time, flux, flux_err):
    
    ### parameter
    t_peak = numpyro.sample('t_peak', dist.Uniform(-100, 100))
    t_rise = numpyro.sample('t_rise', dist.Uniform(.1, 300))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 1000))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(.1, 1000))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.9))
    t_start = t_peak - t_rise   
    a = numpyro.sample('a', dist.Uniform(-1, 10) )
    b = numpyro.sample('b', dist.Uniform(-100, 100) )
    c = numpyro.sample('c', dist.Uniform(-100, 100) )
    d = numpyro.deterministic('d', (f_peak - a * t_rise - b*t_rise**2 - c *t_rise**3)/t_rise**4  )

    time_linear = time - t_start  
    time_dif = time - t_peak
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2
    model_rise  = width_func(time_linear, 0, t_rise) * poly_func_all(time_linear, a, b, c, d) 
    mu = model_rise + jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = jnp.sum(- 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err))
    #loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(model_rise)+0.0000001, 0))

    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)

def model_flare_comp_all_poly_jax(time, flux, flux_err):
    
    ### parameter
    t_peak = numpyro.sample('t_peak', dist.Uniform(-100, 100))
    t_rise = numpyro.sample('t_rise', dist.Uniform(0.1, 10))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 1000))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(0.1, 1000))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.9))
    t_start = t_peak - t_rise   
    a = numpyro.sample('a', dist.Uniform(-1, 10) )
    b = numpyro.sample('b', dist.Uniform(-100, 100) )
    c = numpyro.sample('c', dist.Uniform(-100, 100) )
    d = numpyro.deterministic('d', (f_peak - a * t_rise - b*t_rise**2 - c *t_rise**3)/t_rise**4  )
    time_linear = time - t_start  
    time_dif = time - t_peak
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2
    model_rise  = width_func(time_linear, 0, t_rise) * poly_func_all(time_linear, a, b, c, d) 
    mu = model_rise + jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = jnp.sum(- 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err))
    #loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(model_rise)+0.0000001, 0))
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)

def model_poly(time, flux, flux_err, num_poly):
    
    ### parameter
    lows = np.ones(num_poly)*(-100)
    highs = np.ones(num_poly) * 100
    a = numpyro.sample('a', dist.Uniform(lows, highs))
    mu = jnp.polyval(a,time)
    #loglikelihood = - 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err)
    return numpyro.sample("y", dist.Normal(mu, flux_err), obs=flux)



def model_flare_comp_all_poly_const_jax(time, flux, flux_err):
    
    ### parameter
    t_peak_start = numpyro.sample('t_peak', dist.Uniform(-100, 100))
    t_rise = numpyro.sample('t_rise', dist.Uniform(.1, 300))
    t_peak_duration = numpyro.sample('t_peak_duration', dist.Uniform(0, 100))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 1000))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(.1, 1000))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.9))
    t_start = t_peak_start - t_rise   
    a = numpyro.sample('a', dist.Uniform(0, 10) )
    b = numpyro.deterministic('b', (f_peak - a * t_rise)/t_rise**2  )

    time_linear = time - t_start  
    time_dif = time - (t_peak_start + t_peak_duration)
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2
    poly_model = jnp.polyval(jnp.array([b, a, 0]), time_linear)
    model_rise  = width_func(time_linear, 0, t_rise) * poly_model
    model_peak_const  = width_func(time_linear, t_rise, t_rise + t_peak_duration) * f_peak
    mu = model_rise +model_peak_const +  jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = jnp.sum(- 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(model_rise)+0.0000001, 0))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( b+0.0000001, 0))
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)

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

def care_after_mcmc(time, mcmc,  mcmc_samples, para_names, rng_key_, file_head = ""):
    
    para_dic = make_dirs(para_names,mcmc_samples)
    arviz.plot_trace(mcmc, var_names=["t_rise","t_peak", "f_peak", "t_1_2_decay", 't_ratio', 't_1_2_decay_2', "fraction"])
    plt.savefig(file_head + "_hmc2.pdf")
    plt.close()
    
    arviz.plot_pair(arviz.from_numpyro(mcmc),var_names = para_names, kind='kde',
    divergences=False,marginals=True)
    plt.savefig(file_head + "_posterior.pdf")
    plt.close()

    pred = Predictive(model_flare_comp2_jax,para_dic, return_sites=["mu"])
   # x_dmy = jnp.linspace(np.min(time), np.max(time), 1000)
    null_arr = jnp.ones(len(time))
    predictions = pred(rng_key_,time=time,flux=null_arr, flux_err = null_arr)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    hpdi_muy = hpdi(predictions["mu"], 0.9)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    para_arr = calc_percentile(mcmc_samples, para_names)
    return mean_muy, hpdi_muy, para_arr

def care_after_poly_mcmc(time, mcmc,  mcmc_samples, para_names, rng_key_, file_head = ""):
    
    para_dic = make_dirs(para_names,mcmc_samples)
    arviz.plot_trace(mcmc, var_names=["t_rise","t_peak", "f_peak", "t_1_2_decay", 't_ratio', 't_1_2_decay_2', "fraction"])
    plt.savefig(file_head + "_hmc2.pdf")
    plt.close()
    arviz.plot_pair(arviz.from_numpyro(mcmc),var_names = para_names, kind='kde',
    divergences=False,marginals=True)
    plt.savefig(file_head + "_posterior.pdf")
    plt.close()

    pred = Predictive(model_flare_comp_all_poly_jax, para_dic, return_sites=["mu"])
    #x_dmy = jnp.linspace(np.min(time), np.max(time), 1000)
    null_arr = jnp.ones(len(time))
    predictions = pred(rng_key_,time=time,flux=null_arr, flux_err = null_arr)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    hpdi_muy = hpdi(predictions["mu"], 0.9)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    para_arr = calc_percentile(mcmc_samples, para_names)
    return mean_muy, hpdi_muy, para_arr

def care_after_poly_const_mcmc(time, mcmc_samples, para_names, rng_key_, file_head = ""):
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

    pred = Predictive(model_flare_comp_all_poly_const_jax, para_dic, return_sites=["mu"])
    #x_dmy = jnp.linspace(np.min(time), np.max(time), 1000)
    null_arr = jnp.ones(len(time))
    predictions = pred(rng_key_,time=time,flux=null_arr, flux_err = null_arr)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    hpdi_muy = hpdi(predictions["mu"], 0.9)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    para_arr = calc_percentile(mcmc_samples, para_names)
    return mean_muy, hpdi_muy, para_arr


def run_hmc_flare(time, flux, flux_err,para_names,  num_warmup, num_samples, file_head):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(model_flare_comp2_jax)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, time=time, flux=flux, flux_err =flux_err)
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mean_muy, hpdi_muy, para_arr = care_after_mcmc(time, mcmc, mcmc_samples, para_names, rng_key_, file_head )    
    
    return mean_muy, hpdi_muy, para_arr, mcmc_samples

def run_hmc_flare_polys(time, flux, flux_err,para_names,  num_warmup, num_samples, file_head, value_dic =None):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    print(rng_key_)
    kernel = NUTS(model_flare_comp_all_poly_jax, init_strategy=init_to_value(values=value_dic))

    if  value_dic is not None:
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, time=time, flux=flux, flux_err =flux_err)
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mean_muy, hpdi_muy, para_arr = care_after_poly_mcmc(time, mcmc, mcmc.get_samples(), para_names, rng_key_, file_head )    

    return mean_muy, hpdi_muy, para_arr, mcmc_samples

def run_hmc_flare_polys_peak_const(time, flux, flux_err,para_names,  num_warmup, num_samples, file_head, value_dic =None):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    print(rng_key_)
    kernel = NUTS(model_flare_comp_all_poly_const_jax, init_strategy=init_to_value(values=value_dic))

    if  value_dic is not None:
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, time=time, flux=flux, flux_err =flux_err)
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mean_muy, hpdi_muy, para_arr = care_after_poly_const_mcmc(time, mcmc.get_samples(), para_names, rng_key_, file_head )    
    return mean_muy, hpdi_muy, para_arr, mcmc_samples



def care_after_model_template(time, mcmc_samples, para_names, rng_key_, file_head = ""):
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
    #x_dmy = jnp.linspace(np.min(time), np.max(time), 1000)
    null_arr = jnp.ones(len(time))
    predictions = pred(rng_key_,time=time,flux=null_arr, flux_err = null_arr, num_poly = 4)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    hpdi_muy = hpdi(predictions["mu"], 0.9)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    para_arr = calc_percentile(mcmc_samples, para_names)
    return mean_muy, hpdi_muy, para_arr

def model_flare_for_template_for_return(time, flux, flux_err, para_dic):
    
    ### parameter
    t_peak_start = para_dic["t_peak"]#numpyro.sample('t_peak', dist.Uniform(-1, 1))
    t_rise = para_dic["t_rise"]
    t_peak_duration = para_dic["t_peak_duration"]
    f_peak = para_dic["f_peak"]
    t_1_2_decay =  para_dic["t_1_2_decay"]
    t_ratio = para_dic["t_ratio"]
    t_1_2_decay_2 = t_ratio * t_1_2_decay
    fraction = para_dic["fraction"]
    t_start = t_peak_start - t_rise   
    a = para_dic["a"]
    b =para_dic["b"]
    c = para_dic["c"]
    d =  (a - b + c -1 )

    time_linear = time - t_start 
    time_linear_for_rise = (time - t_peak_start)/t_rise
    time_dif = time - (t_peak_start + t_peak_duration)
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2

    poly_model = np.polyval(f_peak * np.array([d, c, b, a, 1]), time_linear_for_rise )
    model_rise  = width_func_np(time_linear, 0, t_rise) * poly_model# * jnp.exp((time -t_peak_start)/ t_tau_rise)
    model_peak_const  = width_func_np(time_linear, t_rise, t_rise + t_peak_duration) * f_peak
    mu = model_rise +model_peak_const +  np.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  np.exp( -  time_dif/ sigma)   + (1- fraction) * np.exp( -  time_dif/ sigma_2) )

    return mu




def model_flare_4th(time, flux, flux_err, num_poly):
    
    ### parameter
    t_peak_start = numpyro.sample('t_peak', dist.Uniform(-1, 1))
    t_rise = numpyro.sample('t_rise', dist.Uniform(.1, 2))
    t_peak_duration = numpyro.sample('t_peak_duration', dist.Uniform(0, 5))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 2))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(.1, 100))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.99))
    t_start = t_peak_start - t_rise   
    lows = np.ones(num_poly-1)*(-20)
    highs = np.ones(num_poly-1) * 20
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
    numpyro.deterministic('loglike_now', loglikelihood)

def model_flare_for_template(time, flux, flux_err):
    
    ### parameter
    t_peak_start = numpyro.sample('t_peak', dist.Uniform(-1, 1))
    t_rise = numpyro.sample('t_rise', dist.Uniform(.1, 2))
    t_peak_duration = numpyro.sample('t_peak_duration', dist.Uniform(0, 5))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 2))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(.1, 100))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.99))
    t_start = t_peak_start - t_rise   
    a = numpyro.sample('a', dist.Uniform(-20, 20) )
    b = numpyro.sample('b', dist.Uniform(-20, 20) )
    c = numpyro.sample('c', dist.Uniform(-30, 30) )
    d = numpyro.deterministic('d', (a - b + c -1 ))

    time_linear = time - t_start 
    time_linear_for_rise = (time - t_peak_start)/t_rise
    time_dif = time - (t_peak_start + t_peak_duration)
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2

    poly_model = jnp.polyval(f_peak * jnp.array([d, c, b, a, 1]), time_linear_for_rise )
    model_rise  = width_func(time_linear, 0, t_rise) * poly_model# * jnp.exp((time -t_peak_start)/ t_tau_rise)
    model_peak_const  = width_func(time_linear, t_rise, t_rise + t_peak_duration) * f_peak
    mu = model_rise +model_peak_const +  jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = jnp.sum(- 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(model_rise)+0.0000001, 0))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( b+0.0000001, 0))
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)
    numpyro.deterministic('loglike_now', loglike)


def model_flare_for_template_second_order(time, flux, flux_err):
    
    ### parameter
    t_peak_start = numpyro.sample('t_peak', dist.Uniform(-1, 1))
    t_rise = numpyro.sample('t_rise', dist.Uniform(.1, 2))
    t_peak_duration = numpyro.sample('t_peak_duration', dist.Uniform(0, 5))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 2))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(.1, 100))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.99))
    t_start = t_peak_start - t_rise   
    a = numpyro.sample('a', dist.Uniform(-10, 20) )
    b = numpyro.deterministic('b', (a -1 ))

    time_linear = time - t_start 
    time_linear_for_rise = (time - t_peak_start)/t_rise
    time_dif = time - (t_peak_start + t_peak_duration)
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2

    poly_model = jnp.polyval(f_peak * jnp.array([b, a, 1]), time_linear_for_rise )
    model_rise  = width_func(time_linear, 0, t_rise) * poly_model# * jnp.exp((time -t_peak_start)/ t_tau_rise)
    model_peak_const  = width_func(time_linear, t_rise, t_rise + t_peak_duration) * f_peak
    mu = model_rise +model_peak_const +  jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = jnp.sum(- 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(model_rise)+0.0000001, 0))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( b+0.0000001, 0))
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)

def care_after_model_template_second_order(time, mcmc_samples, para_names, rng_key_, file_head = ""):
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

    pred = Predictive(model_flare_for_template_second_order, para_dic, return_sites=["mu"])
    #x_dmy = jnp.linspace(np.min(time), np.max(time), 1000)
    null_arr = jnp.ones(len(time))
    predictions = pred(rng_key_,time=time,flux=null_arr, flux_err = null_arr)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    hpdi_muy = hpdi(predictions["mu"], 0.9)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    para_arr = calc_percentile(mcmc_samples, para_names)

    return mean_muy, hpdi_muy, para_arr
def run_hmc_flare_for_template(time, flux, flux_err,para_names,  num_warmup, num_samples, file_head, value_dic =None):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    print(rng_key_)
    kernel = NUTS(model_flare_4th, init_strategy=init_to_value(values=value_dic), dense_mass=True)
    if  value_dic is not None:
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, time=time, flux=flux, flux_err =flux_err, num_poly = 4)
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mean_muy, hpdi_muy, para_arr = care_after_model_template(time, mcmc.get_samples(), para_names, rng_key_, file_head )    
    return mean_muy, hpdi_muy, para_arr, mcmc_samples

def run_hmc_flare_for_template_second_order(time, flux, flux_err,para_names,  num_warmup, num_samples, file_head, value_dic =None):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    print(rng_key_)
    kernel = NUTS(model_flare_for_template_second_order, init_strategy=init_to_value(values=value_dic))
    if  value_dic is not None:
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, time=time, flux=flux, flux_err =flux_err)
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mean_muy, hpdi_muy, para_arr = care_after_model_template_second_order(time, mcmc.get_samples(), para_names, rng_key_, file_head )    
    return mean_muy, hpdi_muy, para_arr, mcmc_samples


def model_flare_for_template_3_order(time, flux, flux_err):
    
    ### parameter
    t_peak_start = numpyro.sample('t_peak', dist.Uniform(-1, 1))
    t_rise = numpyro.sample('t_rise', dist.Uniform(.2, 1.3))
    t_peak_duration = numpyro.sample('t_peak_duration', dist.Uniform(0, 5))
    f_peak = numpyro.sample('f_peak', dist.Uniform(0, 2))
    t_1_2_decay = numpyro.sample('t_1_2_decay', dist.Uniform(.1, 100))
    t_ratio = numpyro.sample('t_ratio', dist.Uniform(1, 100))
    t_1_2_decay_2 = numpyro.deterministic('t_1_2_decay_2', t_ratio * t_1_2_decay)
    fraction = numpyro.sample('fraction', dist.Uniform(0.1, 0.99))
    a = numpyro.sample('a', dist.Uniform(-10, 20) )
    b = numpyro.sample('b', dist.Uniform(-10, 20) )
    c = numpyro.deterministic('c', (1-a+b ))

    t_start = t_peak_start - t_rise   
    time_linear = time - t_start 
    time_linear_for_rise = (time - t_peak_start)/t_rise
    time_dif = time - (t_peak_start + t_peak_duration)
    sigma = t_1_2_decay
    sigma_2 = t_1_2_decay_2

    poly_model = jnp.polyval(f_peak * jnp.array([c, b, a, 1]), time_linear_for_rise )
    model_rise  = width_func(time_linear, 0, t_rise) * poly_model# * jnp.exp((time -t_peak_start)/ t_tau_rise)
    model_peak_const  = width_func(time_linear, t_rise, t_rise + t_peak_duration) * f_peak
    mu = model_rise +model_peak_const +  jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    loglikelihood = jnp.sum(- 0.5 * (flux- mu)**2/flux_err**2 - jnp.log(flux_err))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( jnp.min(model_rise)+0.0000001, 0))
    loglikelihood +=  - 1000000 * (1- jnp.heaviside( b+0.0000001, 0))
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('mu', mu)

def care_after_model_template_3_order(time, mcmc_samples, para_names, rng_key_, file_head = ""):
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

    pred = Predictive(model_flare_for_template_3_order, para_dic, return_sites=["mu"])
    #x_dmy = jnp.linspace(np.min(time), np.max(time), 1000)
    null_arr = jnp.ones(len(time))
    predictions = pred(rng_key_,time=time,flux=null_arr, flux_err = null_arr)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    hpdi_muy = hpdi(predictions["mu"], 0.9)
    mean_muy = jnp.mean(predictions["mu"], axis=0)
    para_arr = calc_percentile(mcmc_samples, para_names)
    return mean_muy, hpdi_muy, para_arr

def run_hmc_flare_for_template_3_order(time, flux, flux_err,para_names,  num_warmup, num_samples, file_head, value_dic =None):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    print(rng_key_)
    kernel = NUTS(model_flare_for_template_3_order, init_strategy=init_to_value(values=value_dic))
    if  value_dic is not None:
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, time=time, flux=flux, flux_err =flux_err)
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    mean_muy, hpdi_muy, para_arr = care_after_model_template_3_order(time, mcmc.get_samples(), para_names, rng_key_, file_head )    
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
