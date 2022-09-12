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

def model_flare_comp2_after_jax(time, t_peak, t_rise, f_peak, t_1_2_decay,\
                                t_1_2_decay2, fraction):

    t_start = t_peak - t_rise    
    time_linear = time - t_start  
    time_dif = time - t_peak
    sigma = t_1_2_decay/np.log(2)
    sigma_2 = t_1_2_decay_2/np.log(2)
    
    mu = width_func(time_linear, 0, t_rise) * time_linear * f_peak/t_rise\
    + jnp.heaviside(time_dif,0) * f_peak *  \
    ( fraction *  jnp.exp( -  time_dif/ sigma)   + (1- fraction) * jnp.exp( -  time_dif/ sigma_2) )
    return mu

def make_dirs(para_names, samples):
    
    para_dic = {}
    for para_name in para_names:
        para_dic[para_name] = samples[para_name]
    return para_dic

def calc_percentile(mcmc_samples, para_names):
    para_arr = []

    for name in para_names:
        values = np.percentile(mcmc_samples[name], [14, 50, 86])
        para_arr.append(values)

    return para_arr

def linear_trend_jax(x, y, x_err, y_err):
    
    ### parameter
    a0 = numpyro.sample('a0', dist.Uniform(-10, 10))
    b0 = numpyro.sample('b0', dist.Uniform(-30, 30))
    sigma_sum = 2 * (a0**2) * (x_err**2) + 2 * (y_err**2)
    model = x * a0 + b0
    chi = jnp.sum( ((y - model)**2)/sigma_sum)
    loglikelihood = - chi  -  0.5 * jnp.log(sigma_sum)
    numpyro.factor("loglike", loglikelihood)
    numpyro.deterministic('model', model)

def run_linear_trend_jax(x, y, x_err, y_err, num_warmup =2000, num_samples=8000):
    para_names = ["a0", "b0"]
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(linear_trend_jax)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, x=x, y=y, x_err=x_err, y_err=y_err )
    mcmc.print_summary()
    mcmc_samples = mcmc.get_samples()
    para_dic = make_dirs(para_names, mcmc_samples)
    pred = Predictive(linear_trend_jax, para_dic, return_sites=["model"])
    predictions = pred(rng_key_, x=x, y=y, x_err=x_err, y_err=y_err)
    mean_muy = jnp.mean(predictions["model"], axis=0)
    hpdi_muy = hpdi(predictions["model"], 0.9)
    para_arr = calc_percentile(mcmc_samples, para_names)
    return mean_muy, hpdi_muy, para_arr, mcmc_samples


