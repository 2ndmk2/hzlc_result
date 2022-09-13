from hzlc_result import read_output
import lightkurve as lk
import astropy.units as u
import numpy as np
import astropy

def make_lightkurve(time, flux, flux_err):
    time_conv= astropy.time.Time(time, format = 'jd', scale="tdb")
    lc_now = lk.LightCurve()
    lc_now.time = time_conv
    lc_now.flux = flux
    lc_now.flux_err = flux_err
    return lc_now

def make_bined_lc(lc, bin_second):
    lc_binned =  lc.bin(time_bin_size=bin_second * u.second)
    mask = lc_binned.flux != 0 
    return lc_binned[mask]
    
def make_binned_lc(time, flux):
    time_conv= astropy.time.Time(time, format = 'jd', scale="tdb")
    lc_now = lk.LightCurve()
    lc_now.time = time_conv
    lc_now.flux = flux
    
    
def determine_ylim(flux_min, flux_max, wd=0.05):
    flux_width = flux_max - flux_min
    flux_min_lim = flux_min - flux_width *wd
    flux_max_lim = flux_max + flux_width *wd
    return [flux_min_lim, flux_max_lim]

def load_lc_data(path_to_mdwarf_lc_folder, name, date, plot_flux_type, tplotmin, tplotMax, t_rise, t_peak):
    path_to_lc_file = path_to_mdwarf_lc_folder  + str(name) + "/"+ 'light_curve_Gaia-DR2_'  + str(name) +  "_" + str(date) + '.hdf5'
    df = read_output.read_LC_hdf5(path_to_lc_file, date=str(date))
    times_mask, fluxes_mask, flux_errs_mask, flux_pca_mask, mask = read_output.lc_load_from_hdf5(df, plot_flux_name = plot_flux_type)
    tm_second = (times_mask - t_peak)*24.*60*60 * u.second
    tm = (times_mask - t_peak)*24.*60*60     
    flux_ave = np.average(flux_pca_mask[tm < -t_rise])
    relf = (flux_pca_mask - flux_ave)/flux_ave
    relf_errs = flux_errs_mask/flux_ave
    mask = (tm > tplotmin)*(tm < tplotMax)
    time_mask = tm[mask]
    relf_mask = relf [mask]
    relf_errs_mask = relf_errs[mask]
    return time_mask, relf_mask, relf_errs_mask

def plot_lc_for_model_data(time, flux, flux_err, hpdi_muy):


    flux_ylim = determine_ylim(np.min(flux -flux_err), np.max(flux +flux_err))
    fig1 = plt.figure(figsize=figsize)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.ylim(flux_ylim[0], flux_ylim[1])
    plt.errorbar(time,flux,yerr=flux_err,fmt='o',c='k',alpha = 0.75)
    plt.fill_between(time_dmy, hpdi_muy[0], hpdi_muy[1], color="r", alpha=0.5, zorder =100)
    plt.plot(time,flux, c='lightgrey')
    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.ylabel(r"Relative Flux $\ \Delta F/F_{\rm ave}$",fontsize = fontsize_plot )
    plt.yticks(fontsize = fontsize_ticks )

    plt.title('Gaia DR2 '+str(gaia_ids[i])+' / '+ 'TIC '+str(tic_ids[i]), fontsize =fontsize_plot  )

    textstr = '\n'.join((
        r'$M_*=%.2f \,M_\odot$' % (mass[i], ),
        r'$R_*=%.2f \,R_\odot$' % (rad[i], ),
        r'$L_*=%.4f\,L_\odot$' % (Leff[i], ),
        r'$T_\mathrm{eff}=%.0f$ K' % (teff[i], )
    ))

    props = dict(boxstyle='square', facecolor='w', alpha=1)
    frame2=fig1.add_axes((.1,.1,.8,.2))

    plt.errorbar(time,flux -mean_muy ,yerr=flux_err,fmt='o',c='k',alpha = 0.75)


    fn = os.path.join(folder_for_fig, 'GaiaDR2_'+str(gaia_ids[i])+'.pdf')
    plt.xticks(fontsize = fontsize_ticks )
    plt.yticks(fontsize = fontsize_ticks )
    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.savefig(fn, bbox_inches="tight")
    plt.show()
    




