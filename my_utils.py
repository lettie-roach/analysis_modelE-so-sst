#!/usr/bin/python

import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib
import matplotlib.path as mpath
import pandas as pd
import scipy.stats
from scipy.interpolate import splev, splrep
import pickle
xr.set_options(keep_attrs=True)


# customize for own machine
import cartopy
cartopy.config['data_dir'] = "/discover/nobackup/projects/jh_tutorials/JH_examples/JH_datafiles/Cartopy"
cartopy.config['pre_existing_data_dir'] = "/discover/nobackup/projects/jh_tutorials/JH_examples/JH_datafiles/Cartopy"
processed_dir = '/discover/nobackup/laroach1/data_modelE-so-sst/'
cmip_dir = '/css/cmip6/CMIP6/'
obs_dir = '/discover/nobackup/laroach1/OBS/'
figdir = '../figs/'


# convenient global variables
alphabet = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(o)',
           '(p)','(q)','(r)','(s)','(t)','(u)','(v)','(w)','(x)','(y)','(z)']
monthnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
firstofmonthind = [1,32,60,91,121,152,182,213,244,274,305,335]
firstofmonthlabel = ['01-Jan','01-Feb','01-Mar','01-Apr','01-May','01-Jun','01-Jul',
                    '01-Aug','01-Sep','01-Oct','01-Nov','01-Dec']


def linregress(first_samples, second_samples, dim):
    slope, intercept, r_value, p_value, std_err = xr.apply_ufunc(_nanlinregress,
                       first_samples, second_samples,
                       input_core_dims  = [[dim], [dim]], 
                       output_core_dims = [[],[],[],[],[]],
                       vectorize=True)
        
    return slope, intercept, r_value, p_value, std_err

def _nanlinregress(x, y):
    '''Calls scipy linregress only on finite numbers of x and y'''
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        # empty arrays passed to linreg raise ValueError:
        # force returning an object with nans:
        return scipy.stats.linregress([np.nan], [np.nan])
    return scipy.stats.linregress(x[finite], y[finite])


def pearson(first_samples, second_samples, dim):
    pearson, pval = xr.apply_ufunc(_nanpearson,
                       first_samples, second_samples,
                       input_core_dims  = [[dim], [dim]], 
                       output_core_dims = [[],[]],
                       vectorize=True)
        
    return pearson, pval

def _nanpearson(x, y):
    '''Calls scipy linregress only on finite numbers of x and y'''
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        # empty arrays passed to linreg raise ValueError:
        # force returning an object with nans:
        return scipy.stats.pearsonr([np.nan], [np.nan])
    return scipy.stats.pearsonr(x[finite], y[finite])

    
def grid_area_regll(lat,lon):

    to_rad = 2. *np.pi/360.
    r_earth = 6371.22 # km
    con = r_earth*to_rad
    clat = np.cos(lat*to_rad)
    dlon = lon[2] - lon[1]
    dlat = lat[2] - lat[1]
    dx = con*dlon*clat
    dy = con*dlat
    dxdy = dy*dx
    garea = np.swapaxes(np.tile(dxdy,(len(lon),1)),0,1)
    latl = np.swapaxes(np.tile(lat,(len(lon),1)),0,1)
    nh_area = np.where(latl<0.,0.,garea)
    sh_area = np.where(latl>0.,0.,garea)
    
    return garea, nh_area, sh_area


def detrend(first_samples, second_samples, dim):
    
    xdata = np.arange(len(first_samples))
    slope, _, _, _, _ = linregress(xdata, second_samples, dim)
    
    da_x = xr.DataArray(xdata, coords={'time':first_samples[dim]},dims=(dim))
    best_fit = slope*da_x
        
    
    return second_samples - best_fit




def xr_reshape(A, dim, newdims, coords):
    """ Reshape DataArray A to convert its dimension dim into sub-dimensions given by
    newdims and the corresponding coords.
    Example: Ar = xr_reshape(A, 'time', ['year', 'month'], [(2017, 2018), np.arange(12)]) """

    # Create a pandas MultiIndex from these labels
    ind = pd.MultiIndex.from_product(coords, names=newdims)

    # Replace the time index in the DataArray by this new index,
    A1 = A.copy()

    A1.coords[dim] = ind

    # Convert multiindex to individual dims using DataArray.unstack().
    # This changes dimension order! The new dimensions are at the end.
    A1 = A1.unstack(dim)

    # Permute to restore dimensions
    i = A.dims.index(dim)
    dims = list(A1.dims)

    for d in newdims[::-1]:
        dims.insert(i, d)

    for d in newdims:
        _ = dims.pop(-1)


    return A1.transpose(*dims)


def spolar_subplot(lat_max, p1, p2, p3):
    
    ax = plt.subplot(p1, p2, p3, projection = ccrs.SouthPolarStereo())
    ax.set_extent([0.005, 360, lat_max, -90], crs=ccrs.PlateCarree())
    dmeridian = 30  # spacing for lines of meridian
    dparallel = 15  # spacing for lines of parallel 
    num_merid = int(360/dmeridian + 1)
    num_parra = int(90/dparallel + 1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), \
                      xlocs=np.linspace(-180, 180, num_merid), \
                      ylocs=np.linspace(0, -90, num_parra), \
                      linestyle="--", linewidth=1, color='k', alpha=0.5)

    theta = np.linspace(0, 2*np.pi, 120)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    center, radius = [0.5, 0.5], 0.5
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)  #without this; get rect bound
    ax.coastlines("50m")
    #ax.add_feature(cfeature.LAND)
    return ax

def compute_anom(ds, y1 , y2):
    ds = ds.sel(time=slice(str(y1),str(y2)))
    clim = ds.groupby('time.month').mean(dim='time')
    listda = []
    for var in ds:
        myvar = myf.xr_reshape(ds[var], 'time', ['year', 'month'], [np.arange(y1, y2+1,1), np.arange(1,13,1)])
        listda.append(myvar)
    allvar = xr.merge(listda)
    anom = allvar - clim
    anom = anom.stack(time=('year','month'))
    anom['time'] = pd.date_range(start='1/1/'+str(y1), end='1/1/'+str(int(y2)+1),freq='M')
    
    return anom

def plot_prop_from_names (mynames):

    mylist = []
    for f in mynames:
        if 'OBS' in f:
            mylist.append({'label' : 'OBS', 'c' : 'k', 'linewidth' : 2, 'alpha' : 1})
        elif ('f2' in f or 'f1' in f):
            mylist.append({'label' : 'CTRL', 'c' : 'tab:blue', 'linewidth' : .75, 'alpha' : 0.5})
        elif 'f4' in f:
            mylist.append({'label' : 'MW', 'c' : 'tab:green', 'linewidth' : .75, 'alpha' : 0.5})
        elif 'f5' in f:
            mylist.append({'label' : 'WIND', 'c' : 'tab:red', 'linewidth' : 2, 'alpha' : 1})
        elif 'f6' in f:
            mylist.append({'label' : 'WIND&MW', 'c' : 'orange', 'linewidth' : 2, 'alpha' : 1})

        elif 'CTRLmean' in f:
            mylist.append({'label' : 'CTRLmean', 'c' : 'tab:blue', 'linewidth' : 3, 'alpha' : 1})
        elif 'MWmean' in f:
            mylist.append({'label' : 'MWmean', 'c' : 'tab:green', 'linewidth' : 3, 'alpha' : 1})
        elif 'WINDmean' in f:
            mylist.append({'label' : 'WINDmean', 'c' : 'tab:red', 'linewidth' : 3, 'alpha' : 1})
        elif 'WIND&MWmean' in f:
            mylist.append({'label' : 'WIND&MWmean', 'c' : 'orange', 'linewidth' : 3, 'alpha' : 1})
        else:
            print('Need new case for '+f)

    df = pd.DataFrame(mylist, index = mynames) 
    df = df.transpose()

    return df