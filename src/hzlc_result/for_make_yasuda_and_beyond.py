"""
Modules to make yasuda plots

Author: masa
"""

from lightkurve import search_targetpixelfile
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import read_output
from astropy.time import Time
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import os
import load_data


def take_gaia_stars(ra_now, dec_now):
    coord = SkyCoord(ra=ra_now, dec=dec_now, unit=(u.degree, u.degree), frame='icrs')
    radius = u.Quantity(100.0, u.arcsec)
    j = Gaia.cone_search_async(coord, radius)
    r = j.get_results()
    r_dist = r["dist"]/((1/3600))
    source_id = r["source_id"]
    phot_g_mean_mag= r["phot_g_mean_mag"]
    return source_id, phot_g_mean_mag, r_dist 

def make_movie_mp4(movie_bin, file_name = "anim.mp4"):
    # show movie
    nt, nx, ny = np.shape(movie_bin)
    vlims = np.percentile(movie_bin, [1,99])
    fig, ax = plt.subplots( )
    im = ax.imshow(movie_bin[0,:,:],vmin = vlims[0], vmax = vlims[1])
    def init():
        im.set_data(movie_bin[0,:,:])
        return (im,)

    # animation function. This is called sequentially
    def animate(i):
        data_slice = movie_bin[i,:,:]
        im.set_data(data_slice)
        return (im,)

    anim = FuncAnimation(fig, animate, init_func=init, interval=30, frames=int(nt), blit=True)
    HTML(anim.to_html5_video())
    anim.save(file_name, dpi=150, fps = 30, writer='ffmpeg',extra_args=['-vcodec', 'libx264'])
    
def lc_make(movie, r_pix = 5):
    nt, nx, ny = np.shape(movie)
    x = np.arange(nx)
    y = np.arange(ny)
    x = x - np.mean(x)
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x,x)
    r_dist = (xx**2 + yy**2)**0.5
    flux = np.sum(movie[:, r_dist<r_pix ], axis = (1))
    return flux

def number_of_pix_aperture(movie, r_pix = 7):
    nt, nx, ny = np.shape(movie)
    x = np.arange(nx)
    y = np.arange(ny)
    x = x - np.mean(x)
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x,x)
    r_dist = (xx**2 + yy**2)**0.5
    numer_of_pix = len(movie[0, r_dist<r_pix ])
    return numer_of_pix 

def back_movie(movie, r_pix = 7):
    nt, nx, ny = np.shape(movie)
    x = np.arange(nx)
    y = np.arange(ny)
    x = x - np.mean(x)
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x,x)
    r_dist = (xx**2 + yy**2)**0.5
    r_aperture_pix = len(r_dist<r_pix )
    flux = np.median(movie[:, r_dist>r_pix ], axis = 1)
    return flux 

def back_estimate(movie, r_pix = 5):
    nt, nx, ny = np.shape(movie)
    x = np.arange(nx)
    y = np.arange(ny)
    x = x - np.mean(x)
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x,x)
    r_dist = (xx**2 + yy**2)**0.5
    r_aperture_pix = len(r_dist<r_pix )
    flux = np.median(movie[:, r_dist>r_pix ], axis = (1))
    return flux * r_aperture_pix

def centroid_estimate(movie):
    nt, nx, ny = np.shape(movie)
    x = np.arange(nx)
    y = np.arange(ny)
    x = x - np.mean(x)
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x,x)
    flux_sum = np.sum(movie, axis=(1,2))
    x_cen = np.einsum("ijk,jk->i",movie, xx)/flux_sum
    y_cen = np.einsum("ijk,jk->i",movie, yy)/flux_sum
    return x_cen, y_cen 



def movie_integrate(times, movie, count_num=10):
    
    
    time_now = np.arange(0,len(movie),count_num)
    movie_bin =[]
    time_bin =[]

    for i in range(len(time_now)-1):
        time_bin.append(np.median(times[time_now[i]:time_now[i+1]]))
        movie_bin.append(np.median(movie[time_now[i]:time_now[i+1]], axis=0))
    return np.array(time_bin), np.array(movie_bin)
    

def make_lc_plot_and_snapshot(time, lc, images, n_snap, time_range):
    arg_int_for_lc = np.arange(len(lc))
    mask = (time > time_range[0]) *  (time < time_range[1])
    dx = (np.max(arg_int_for_lc[mask])- np.min(arg_int_for_lc[mask]))/(n_snap)
    index_for_movie = np.arange(0.5*dx + np.min(arg_int_for_lc[mask]), 0.5*dx + np.max(arg_int_for_lc[mask]), dx )
    index_for_movie = index_for_movie.astype(int)
    vmax = np.max(images[mask])
    vmin = np.min(images[mask])
    
    fig = plt.figure(figsize=(n_snap*0.7*1.5, n_snap*0.5 * 1.5))
    
    plt.subplots_adjust(wspace= 0., hspace= 0.)
    for i in range(n_snap):
        ax_now = plt.subplot2grid((5,n_snap), (0,i)) 
        #sub2 = fig.add_subplot(2,n_snap,i+1)
        ax_now.imshow(images[index_for_movie[i]], vmin=vmin, vmax = vmax)
        plt.xticks([])
        plt.yticks([])

    #sub1 = fig.add_subplot(2,1,2)
    ax_now = plt.subplot2grid((5,n_snap), (1,0), colspan=n_snap, rowspan=4) 

    plt.plot(time[mask], lc[mask], color = "k")
    plt.xlim(np.min(time[mask]), np.max(time[mask]))
    plt.xlabel("time (s)", fontsize =n_snap * 1.5)
    plt.ylabel("flux", fontsize =n_snap * 1.5)
    plt.scatter(time[index_for_movie], lc[index_for_movie],s = 50,  color="r", zorder=100)
    #plt.scatter(time[index_for_movie], lc[index_for_movie], color="r")

    plt.savefig('movie_lc.pdf', dpi = 100, bbox_inches = 'tight')    
    

    
    

def make_lc_plot_and_snapshot_modified(time, lc, time_bin, image_bin, n_snap, time_range, bkg = None):
    arg_int_for_lc = np.arange(len(lc))
    mask = (time > time_range[0]) *  (time < time_range[1])
    dx = (np.max(arg_int_for_lc[mask])- np.min(arg_int_for_lc[mask]))/(n_snap)
    index_for_movie = np.arange(0.5*dx + np.min(arg_int_for_lc[mask]), 0.5*dx + np.max(arg_int_for_lc[mask]), dx )
    time_mask_for_bin = (time_bin > time_range[0]) *  (time_bin < time_range[1]) 
    index_for_movie = index_for_movie.astype(int)
    lim_masked_image = image_bin[time_mask_for_bin][np.isfinite(image_bin[time_mask_for_bin])]
    vmax = np.max(lim_masked_image)
    vmin = np.min(lim_masked_image)
    
    fig = plt.figure(figsize=(n_snap*0.7*1.5, n_snap*0.5 * 1.5))
    
    plt.subplots_adjust(wspace= 0., hspace= 0.)
    for i in range(n_snap):
        time_now = time[index_for_movie[i]]
        ax_now = plt.subplot2grid((5,n_snap), (0,i)) 
        #sub2 = fig.add_subplot(2,n_snap,i+1)
        time_index = np.argmin( (time_now - time_bin)**2)
        ax_now.imshow(image_bin[time_index ], vmin=vmin, vmax = vmax*0.8)
        plt.xticks([])
        plt.yticks([])

    #sub1 = fig.add_subplot(2,1,2)
    ax_now = plt.subplot2grid((5,n_snap), (1,0), colspan=n_snap, rowspan=4) 
    ax2=ax_now.twinx()


    ax_now.plot(time[mask], lc[mask],  label="target", color="k")
    ax_now.set_xlim(np.min(time[mask]), np.max(time[mask]))
    ax_now.set_xlabel("time (s)", fontsize =n_snap * 1.5)
    ax_now.set_ylabel("flux", fontsize =n_snap * 1.5)
    #plt.scatter(time[index_for_movie], lc[index_for_movie],s = 50,  color="r", zorder=100)
    ax_now.scatter(time[index_for_movie], lc[index_for_movie], color="r", zorder=100)

    if bkg is not None:
        bkg_time =  bkg[0][mask]
        bkg_masked = bkg[1][mask]
        ax2.plot(bkg_time ,bkg_masked, label="background", color="b")
        mask_bkg = np.isfinite(bkg_masked)
        wd = np.max(bkg_masked[mask_bkg]) - np.min(bkg_masked[mask_bkg])
        y_min_bkg = np.min(bkg_masked[mask_bkg]) - 0.1*wd
        y_max_bkg = np.max(bkg_masked[mask_bkg]) + 0.1*wd
        ax2.set_ylim(y_min_bkg, y_max_bkg)
        ax2.set_ylabel("Background", fontsize =n_snap * 1.5)

    plt.legend(fontsize = n_snap * 1.5)
    plt.savefig('movie_lc.pdf', dpi = 100, bbox_inches = 'tight')    
        
def yasuda_plot_wth_centroids(time, lc, time_bin, image_bin, n_snap, time_range, center_x, center_y):
    #arg_tmp = (time > time_range[0]) *  (time < time_range[1])
    arg_int_for_lc = np.arange(len(lc))
    mask = (time > time_range[0]) *  (time < time_range[1])


    dx = int((np.max(arg_int_for_lc[mask])- np.min(arg_int_for_lc[mask]))/(n_snap))

    dx = (np.max(arg_int_for_lc[mask])- np.min(arg_int_for_lc[mask]))/(n_snap)    
    index_for_movie = np.arange(0.5*dx + np.min(arg_int_for_lc[mask]), 0.5*dx + np.max(arg_int_for_lc[mask]), dx )
    time_mask_for_bin = (time_bin > time_range[0]) *  (time_bin < time_range[1]) 
    index_for_movie = index_for_movie.astype(int)
    lim_masked_image = image_bin[time_mask_for_bin][np.isfinite(image_bin[time_mask_for_bin])]
    vmax = np.max(lim_masked_image)
    vmin = np.min(lim_masked_image)
    
    fig = plt.figure(figsize=(n_snap*0.7*1.5, n_snap*0.5 * 1.5))
    
    plt.subplots_adjust(wspace= 0., hspace= 0.)
    for i in range(n_snap):
        time_now = time[index_for_movie[i]]
        ax_now = plt.subplot2grid((5,n_snap), (0,i)) 
        #sub2 = fig.add_subplot(2,n_snap,i+1)
        time_index = np.argmin( (time_now - time_bin)**2)
        ax_now.imshow(image_bin[time_index ], vmin=vmin, vmax = vmax*0.8)
        plt.xticks([])
        plt.yticks([])

    #sub1 = fig.add_subplot(2,1,2)
    ax_now = plt.subplot2grid((5,n_snap), (1,0), colspan=n_snap, rowspan=4) 
    ax2=ax_now.twinx()
    
    ax2.plot(time, center_x, color="r")
    ax2.plot(time, center_y, color="b")
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("centroid", fontsize =n_snap * 1.5)

    ax_now.plot(time[mask], lc[mask], color = "k")
    ax_now.set_xlim(np.min(time[mask]), np.max(time[mask]))
    ax_now.set_xlabel("time (s)", fontsize =n_snap * 1.5)
    ax_now.set_ylabel("flux", fontsize =n_snap * 1.5)
    
    #plt.scatter(time[index_for_movie], lc[index_for_movie],s = 50,  color="r", zorder=100)
    plt.scatter(time[index_for_movie], lc[index_for_movie], color="r")

    plt.savefig('movie_lc.pdf', dpi = 100, bbox_inches = 'tight')    
    

def time_return(time_min, time_max, times, t_peak_id, dx):
    
    time_arr = []
    t_now = t_peak_id
    while(1):
        t_now = t_now- dx
        if t_now < 0:
            break
        if not times[t_now] < time_min:
            if t_now < 0:
                break
            time_arr.append(t_now)
        else:
            break
            
    time_arr.append(t_peak_id)
    t_now = t_peak_id
    while(1):
        t_now = t_now + dx
        if t_now > len(times)-1:
            break        
        if not times[t_now]> time_max:
            if t_now > len(times)-1:
                break                
            time_arr.append(t_now)
        else:
            break
    time_arr = np.array(time_arr)
    time_arr = np.sort(time_arr)
    return time_arr


def make_lc_plot_and_snapshot_modified_tmp(time, lc, time_bin, image_bin, n_snap, time_range, t_peak, file = "movie_lc.pdf"):
    arg_int_for_lc = np.arange(len(lc))
    mask_tmp = (time > time_range[0]) *  (time < time_range[1])
    t_peak_id = time_index = np.argmin( (time- t_peak)**2)
    dx = int((np.max(arg_int_for_lc[mask_tmp])- np.min(arg_int_for_lc[mask_tmp]))/(n_snap))
    index_for_movie = time_return(time_range[0], time_range[1], time, t_peak_id, dx)
    time_max_id = np.min([int(np.max(index_for_movie) + 0.5 * dx), len(time)-1])
    time_min_id = np.max([0, int(np.min(index_for_movie) - 0.5 * dx)])

    mask = (time > time[time_min_id]) * (time < time[time_max_id])
    time_mask_for_bin = (time_bin > time_range[0]) *  (time_bin < time_range[1]) 
    lim_masked_image = image_bin[time_mask_for_bin][np.isfinite(image_bin[time_mask_for_bin])]
    vmax = np.max(lim_masked_image)
    vmin = np.min(lim_masked_image)
    n_snap_now = len(index_for_movie)
    fig = plt.figure(figsize=( n_snap_now*0.7*1.5, n_snap*0.5 * 1.5))
    
    plt.subplots_adjust(wspace= 0., hspace= 0.)
    for i in range(n_snap_now ):
        time_now = time[index_for_movie[i]]
        ax_now = plt.subplot2grid((5,n_snap_now ), (0,i)) 
        #sub2 = fig.add_subplot(2,n_snap,i+1)
        time_index = np.argmin( (time_now - time_bin)**2)
        ax_now.imshow(image_bin[time_index ], vmin=vmin, vmax = vmax*0.8)
        plt.xticks([])
        plt.yticks([])

    #sub1 = fig.add_subplot(2,1,2)
    ax_now = plt.subplot2grid((5,n_snap_now ), (1,0), colspan=n_snap_now , rowspan=4) 

    plt.plot(time[mask], lc[mask], color = "k")
    plt.xlim(np.min(time[mask]), np.max(time[mask]))
    plt.xlabel("time (s)", fontsize =n_snap_now  * 1.5)
    plt.ylabel("flux", fontsize =n_snap_now  * 1.5)
    plt.xticks(fontsize = 23)
    plt.yticks(fontsize = 23)
    #plt.scatter(time[index_for_movie], lc[index_for_movie],s = 50,  color="r", zorder=100)
    print(len(time), len(lc))
    #print(index_for_movie)
    #print(time)
    #plt.scatter(time[np.array(index_for_movie)], lc[np.array(index_for_movie)], color="r", zorder=100)

    plt.savefig(file, dpi = 100, bbox_inches = 'tight')  
    plt.show()


def make_lc_plot_for_papers(time, lc, time_bin, image_bin, n_snap, time_range, t_peak, file = "movie_lc.pdf"):
    arg_int_for_lc = np.arange(len(lc))
    mask_tmp = (time > time_range[0]) *  (time < time_range[1])
    t_peak_id = time_index = np.argmin( (time- t_peak)**2)
    dx = int((np.max(arg_int_for_lc[mask_tmp])- np.min(arg_int_for_lc[mask_tmp]))/(n_snap))
    index_for_movie = time_return(time_range[0], time_range[1], time, t_peak_id, dx)
    time_max_id = np.min([int(np.max(index_for_movie) + 0.5 * dx), len(time)-1])
    time_min_id = np.max([0, int(np.min(index_for_movie) - 0.5 * dx)])

    mask = (time > time[time_min_id]) * (time < time[time_max_id])
    time_mask_for_bin = (time_bin > time_range[0]) *  (time_bin < time_range[1]) 
    lim_masked_image = image_bin[time_mask_for_bin][np.isfinite(image_bin[time_mask_for_bin])]
    vmax = np.max(lim_masked_image)
    vmin = np.min(lim_masked_image)
    n_snap_now = len(index_for_movie)
    fig = plt.figure(figsize=( n_snap_now*0.6*1.5, n_snap*0.45 * 1.5))
    
    plt.subplots_adjust(wspace= 0., hspace= 0.)
    time_arr = []
    for i in range(n_snap_now ):
        time_now = time[index_for_movie[i]]
        time_arr .append(time_now)


        ax_now = plt.subplot2grid((5,n_snap_now ), (0,i)) 
        #sub2 = fig.add_subplot(2,n_snap,i+1)
        time_index = np.argmin( (time_now - time_bin)**2)
        ax_now.imshow(image_bin[time_index ], vmin=vmin, vmax = vmax*0.8)
        plt.xticks([])
        plt.yticks([])


    #sub1 = fig.add_subplot(2,1,2)
    ax_now = plt.subplot2grid((5,n_snap_now ), (1,0), colspan=n_snap_now , rowspan=4) 
    time_sec = (time[mask] )* 3600*24
    time_for_movie = (time[index_for_movie])* 3600*24
    plt.plot(time_sec, lc[mask], color = "k")
    plt.xlim(np.min(time_sec), np.max(time_sec))
    plt.xlabel("Time [sec]", fontsize =23)
    plt.ylabel("Relative Flux", fontsize =23)
    plt.xticks(fontsize = 23)
    plt.yticks(fontsize = 23)
    plt.rcParams['axes.titley'] = 1.22
    plt.title("Zoom-up view", fontsize = 23)
    #plt.scatter(time[index_for_movie], lc[index_for_movie],s = 50,  color="r", zorder=100)
    #plt.scatter(time_for_movie , lc[index_for_movie], color="r", zorder=100)
    print(np.min(time_sec), np.max(time_sec))
    plt.savefig(file, dpi = 100, bbox_inches = 'tight')  
    plt.show()
    return time_arr

    

def make_diff_image(movie, t_peak_id, delta_t_id):
    bright_image = np.mean(movie[t_peak_id - delta_t_id: t_peak_id + delta_t_id], axis=0)
    normal_image = np.mean(movie[t_peak_id - 2 * delta_t_id: t_peak_id - delta_t_id], axis=0)
    diff_image = bright_image - normal_image
    return bright_image, normal_image, diff_image 


    
