from hzlc_result import read_output
import lightkurve as lk
import astropy.units as u
import numpy as np
import astropy
from hzlc_result import lc_operation
import matplotlib.pyplot as plt
import os

def plot_lc_for_model_data(time, flux, flux_err, hpdi_muy, mean_muy, folder_for_fig, tic_id, zoom = False, file_name = 'comp_with_model.pdf', mass =None, hpdi_muy_20 = None, xlims = None,zoom_xlim = None):

    if not os.path.exists(folder_for_fig):
        os.makedirs(folder_for_fig)

    fontsize_plot = 32
    #fontsize_plot = 28
    fontsize_legend = 24
    fontsize_ticks = 24
    figsize = (14, 10)
    try_20 = False

    flux_ylim = lc_operation.determine_ylim(np.min(flux -flux_err), np.max(flux +flux_err))
    fig1 = plt.figure(figsize=figsize)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.ylim(flux_ylim[0], flux_ylim[1])
    plt.errorbar(time,flux,yerr=flux_err,fmt='o',c='k',alpha = 0.75)
    plt.fill_between(time, hpdi_muy[0], hpdi_muy[1], color="r", alpha=0.5, zorder =100)
    if hpdi_muy_20 is not None:
        plt.fill_between(time_20, hpdi_muy_20[0], hpdi_muy_20[1], color="b", alpha=0.5, zorder =110)
    if not zoom :
        plt.plot(time,flux, c='lightgrey')
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])

    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.ylabel(r"Relative Flux $\ \Delta F/F_{\rm ave}$",fontsize = fontsize_plot )
    plt.yticks(fontsize = fontsize_ticks )
    if zoom_xlim is not None:
        plt.vlines(zoom_xlim[0], -100, 100, color='b', linestyles='dotted', lw = 3)
        plt.vlines(zoom_xlim[1], -100, 100, color='b', linestyles='dotted', lw = 3)


    if zoom:
        plt.title('TIC '+str(tic_id) + " (zoom)" , fontsize =fontsize_plot  )
    else:

        if mass is None:
            plt.title('TIC '+str(tic_id) , fontsize =fontsize_plot  )
        else:
            plt.title('TIC '+str(tic_id) +  " $(%.2f M_{\odot})$" % mass  , fontsize =fontsize_plot  )
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.errorbar(time,flux -mean_muy ,yerr=flux_err,fmt='o',c='k',alpha = 0.75)
    fn = os.path.join(folder_for_fig, 'TIC_'+str(tic_id)+file_name )
    plt.xticks(fontsize = fontsize_ticks )
    plt.yticks(fontsize = fontsize_ticks )
    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.ylabel("Residual",fontsize = fontsize_plot )
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])

    """
    textstr = '\n'.join((
        r'$M_*=%.2f \,M_\odot$' % (mass[i], ),
        r'$R_*=%.2f \,R_\odot$' % (rad[i], ),
        r'$L_*=%.4f\,L_\odot$' % (Leff[i], ),
        r'$T_\mathrm{eff}=%.0f$ K' % (teff[i], )
    ))

    props = dict(boxstyle='square', facecolor='w', alpha=1)
    """
    plt.savefig(fn, bbox_inches="tight")
    plt.show()

def plot_lc_for_model_data_zoom(time, flux, flux_err, hpdi_muy, mean_muy, folder_for_fig, tic_id, zoom = False, file_name = 'comp_with_model.pdf', mass =None, hpdi_muy_20 = None, xlims = None,zoom_xlim = None):

    if not os.path.exists(folder_for_fig):
        os.makedirs(folder_for_fig)

    fontsize_plot = 32
    #fontsize_plot = 28
    fontsize_legend = 24
    fontsize_ticks = 24
    figsize = (14, 10)
    try_20 = False

    flux_ylim = lc_operation.determine_ylim(np.min(flux -flux_err), np.max(flux +flux_err))
    fig1 = plt.figure(figsize=figsize)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.ylim(flux_ylim[0], flux_ylim[1])
    plt.errorbar(time,flux,yerr=flux_err,fmt='o',c='k',alpha = 0.75)
    plt.fill_between(time, hpdi_muy[0], hpdi_muy[1], color="r", alpha=0.5, zorder =100)
    if hpdi_muy_20 is not None:
        plt.fill_between(time_20, hpdi_muy_20[0], hpdi_muy_20[1], color="b", alpha=0.5, zorder =110)
    plt.plot(time,flux, c='lightgrey')
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])

    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.ylabel(r"Relative Flux $\ \Delta F/F_{\rm ave}$",fontsize = fontsize_plot )
    plt.yticks(fontsize = fontsize_ticks )
    if zoom_xlim is not None:
        plt.vlines(zoom_xlim[0], -100, 100, color='b', linestyles='dotted', lw = 3)
        plt.vlines(zoom_xlim[1], -100, 100, color='b', linestyles='dotted', lw = 3)


    if zoom:
        plt.title('TIC '+str(tic_id) + " (zoom)" , fontsize =fontsize_plot  )
    else:

        if mass is None:
            plt.title('TIC '+str(tic_id) , fontsize =fontsize_plot  )
        else:
            plt.title('TIC '+str(tic_id) +  " $(%.2f M_{\odot})$" % mass  , fontsize =fontsize_plot  )
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.errorbar(time,flux -mean_muy ,yerr=flux_err,fmt='o',c='k',alpha = 0.75)
    fn = os.path.join(folder_for_fig, 'TIC_'+str(tic_id)+file_name )
    plt.xticks(fontsize = fontsize_ticks )
    plt.yticks(fontsize = fontsize_ticks )
    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.ylabel("Residual",fontsize = fontsize_plot )
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])

    """
    textstr = '\n'.join((
        r'$M_*=%.2f \,M_\odot$' % (mass[i], ),
        r'$R_*=%.2f \,R_\odot$' % (rad[i], ),
        r'$L_*=%.4f\,L_\odot$' % (Leff[i], ),
        r'$T_\mathrm{eff}=%.0f$ K' % (teff[i], )
    ))

    props = dict(boxstyle='square', facecolor='w', alpha=1)
    """
    plt.savefig(fn, bbox_inches="tight")
    plt.show()

def plot_lc_for_TIC3585(time, flux, flux_err, hpdi_muy, mean_muy, file_name, xlim, title, mass =None, hpdi_muy_20 = None):

    if not os.path.exists(folder_for_fig):
        os.makedirs(folder_for_fig)

    fontsize_plot = 32
    #fontsize_plot = 28
    fontsize_legend = 24
    fontsize_ticks = 24
    figsize = (14, 10)
    try_20 = False

    flux_ylim = lc_operation.determine_ylim(np.min(flux -flux_err), np.max(flux +flux_err))
    fig1 = plt.figure(figsize=figsize)
    frame1=fig1.add_axes((.1,.6,.8,.5))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(flux_ylim[0], flux_ylim[1])
    plt.errorbar(time,flux,yerr=flux_err,fmt='o',c='k',alpha = 0.75)
    plt.fill_between(time, hpdi_muy[0], hpdi_muy[1], color="r", alpha=0.5, zorder =100)
    if hpdi_muy_20 is not None:
        plt.fill_between(time_20, hpdi_muy_20[0], hpdi_muy_20[1], color="b", alpha=0.5, zorder =110)
    plt.plot(time,flux, c='lightgrey')

    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.ylabel(r"Relative Flux $\ \Delta F/F_{\rm ave}$",fontsize = fontsize_plot )
    plt.yticks(fontsize = fontsize_ticks )
    plt.title(title , fontsize =fontsize_plot  )
    frame2=fig1.add_axes((.1,.1,.8,.5))
    plt.xlim(xlim[0], xlim[1])
    plt.errorbar(time,flux -mean_muy ,yerr=flux_err,fmt='o',c='k',alpha = 0.75)
    plt.xticks(fontsize = fontsize_ticks )
    plt.yticks(fontsize = fontsize_ticks )
    plt.xlabel(r"Time from Flare Peak $\ t - t_{\rm peak} $ [sec] ",fontsize = fontsize_plot )
    plt.ylabel("Residual",fontsize = fontsize_plot )

    """
    textstr = '\n'.join((
        r'$M_*=%.2f \,M_\odot$' % (mass[i], ),
        r'$R_*=%.2f \,R_\odot$' % (rad[i], ),
        r'$L_*=%.4f\,L_\odot$' % (Leff[i], ),
        r'$T_\mathrm{eff}=%.0f$ K' % (teff[i], )
    ))

    props = dict(boxstyle='square', facecolor='w', alpha=1)
    """
    plt.savefig(file_name, bbox_inches="tight")
    plt.show()


