
import numpy as np
#https://arxiv.org/abs/1307.2657

def determine_spec(teff):

    Teff_arr = [4200, 4050, 3970, 3880, 3850, 3680, 3550, 3400, 3200, 3050, 2800, 2650, 2570, 2450]
    star_arr = ["K6V", "K7V", "K8V","K9V", "M0V", "M1V", "M2V", "M3V","M4V","M5V", "M6V", "M7V", "M8V", "M9V"]
    idx = np.abs(np.asarray(Teff_arr ) - teff).argmin()
    return star_arr[idx]


def determine_spec_arr(teff_arr):

    spec_arr = []
    for teff_now in teff_arr:
        spec_arr.append(determine_spec(teff_now ))
    return np.array(spec_arr)