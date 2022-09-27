from hzlc_result import read_output
import lightkurve as lk
import astropy.units as u
import numpy as np
import astropy
from hzlc_result import lc_operation
import matplotlib.pyplot as plt
import os
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from hzlc_result import load_data

def plot_lc_for_model_data(time, flux, flux_err, hpdi_muy, mean_muy, folder_for_fig, tic_id, zoom = False, file_name = 'comp_with_model.pdf', mass =None, hpdi_muy_20 = None, xlims = None,zoom_xlim = None):

    if not os.path.exists(folder_for_fig):
        os.makedirs(folder_for_fig)

    fontsize_plot = 32
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
def plot_HR_diagram(df_catalog_target):
    list_of_gabs, list_of_gbp_grp = get_gmag_and_color(df_catalog_target)

    fig = plt.figure(figsize=(8, 8)) 
    ax = fig.add_subplot(111) 
    ax.set(xlabel='$G_{BP}-G_{RP}$', ylabel='$G_{abs} [mag]$', title='GaiaDR2 HR diagram') 
    ax.set_xlim(-1,5)
    ax.set_ylim(16,-3)

    plt.scatter(list_of_gbp_grp, list_of_gabs, alpha = 0.5)
    plt.grid(True)
    plt.show()
    
def plot_skymap_equatorial(df_catalog_target):
    ra_target = np.array(df_catalog_target['ra'])*u.degree
    dec_target = np.array(df_catalog_target['dec'])*u.degree
    c_target = SkyCoord(ra=ra_target, dec=dec_target, frame='icrs')
    
    ra_rad_target = c_target.ra.wrap_at(180*u.deg).radian
    dec_rad_target = c_target.dec.radian
        
    plt.figure(figsize=(15,7.5))
    plt.subplot(111, projection="mollweide")
    plt.grid(True,linestyle='dotted')
    plt.scatter(ra_rad_target, dec_rad_target, alpha=0.5)
    plt.subplots_adjust(top=0.95,bottom=0.0)
    
    gp = SkyCoord(l = np.linspace(-62.5, 295,100)*u.degree, b= np.linspace(0, 0, 100)*u.degree, frame='galactic')
    gp_icrs = gp.transform_to('icrs')
    ra_gp = gp_icrs.ra.wrap_at(180*u.deg).radian
    dec_gp = gp_icrs.dec.radian
    plt.title('Equatorial coordinte')
    plt.plot(ra_gp,dec_gp,c='gray',linestyle = 'dashed')
    plt.show()

    
def plot_skymap_galactic(df_catalog_target):
    ra_target = np.array(df_catalog_target['ra'])*u.degree
    dec_target = np.array(df_catalog_target['dec'])*u.degree
    c_target = SkyCoord(ra=ra_target, dec=dec_target, frame='icrs')
    c_target = c_target.transform_to('galactic')

    ra_rad_target = c_target.l.wrap_at(180*u.deg).radian
    dec_rad_target = c_target.b.radian

    plt.figure(figsize=(15,7.5))
    plt.subplot(111, projection="mollweide")
    plt.grid(True,linestyle='dotted')
    plt.scatter(ra_rad_target, dec_rad_target, alpha=0.5)
    plt.subplots_adjust(top=0.95,bottom=0.0)

    gp = SkyCoord(l = np.linspace(-180, 180, 128)*u.degree, b= np.linspace(0, 0, 128)*u.degree, frame='galactic')
    plt.title('Galactic coordinte')
    plt.plot(gp.l.wrap_at(180*u.deg).radian,gp.b.radian,c='gray',linestyle='dotted')
    plt.show()
