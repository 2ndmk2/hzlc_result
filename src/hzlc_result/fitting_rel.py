import lmfit as lf
import lmfit.models as lfm
def func_linear(x, a0, b0):
    return (x-a0)/b0 


def func_fitting(x, y, wt):
    par_val = {
        'a0' : 80.20784139  ,
        'b0' : 2.54948863,
    }

    par_min = { 
        'a0' : 0,
        'b0' : 0,
    }

    par_max = { 
        'a0' : 300,
        'b0' : 10,
    }

    par_vary = { 
        'a0' : True,
        'b0' : True,
    }

    model = lf.Model(func_linear) #+  lfm.GaussianModel(prefix='gauss1_') + lfm.GaussianModel(prefix='gauss2_')

    params = model.make_params()
    for name in model.param_names: 
        params[name].set(
            value=par_val[name], 
            min=par_min[name],
            max=par_max[name], 
            vary=par_vary[name] 
        )
    result = model.fit(x=x, data=y, weights= wt, params=params, method='leastsq')

    print(result.fit_report())
    #func_now = lambda x: 
    return result
