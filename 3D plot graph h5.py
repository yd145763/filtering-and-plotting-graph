# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:03:13 2023

@author: limyu
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import time


horizontal_peaks = []
horizontal_peaks_position = []
horizontal_peaks_max = []
horizontal_half = []
horizontal_full = []

verticle_peaks = []
verticle_peaks_position = []
verticle_peaks_max = []
verticle_half = []
verticle_full = []

filename = ["grating012umpitch05dutycycle15um", "grating012umpitch05dutycycle20um"]
link = ["C:\\Users\\limyu\\Downloads\grating012umpitch05dutycycle15um.h5", "C:\\Users\\limyu\\Downloads\grating012umpitch05dutycycle20um.h5"]

# Load the h5 file
with h5py.File("C:\\Users\\limyu\\Downloads\grating012umpitch05dutycycle20um.h5", 'r') as f:
    # Get the dataset
    dset = f['grating012umpitch05dutycycle20um']
    # Load the dataset into a numpy array
    arr_3d_loaded = dset[()]



x = np.linspace(-20e-6, 80e-6, num=1950)
y = np.linspace(-25e-6, 25e-6, num = 975)
z = np.linspace(-5e-6, 45e-6, num = 317)

N = np.arange(0, 317, 1)
for n in N:
    print(n, z[n])
    df = arr_3d_loaded[:,:,n]
    df = pd.DataFrame(df)
    df = df.transpose()
    max_E_field = df.max().max()
    row, col = np.where(df == max_E_field)
    row = int(float(row[0]))
    col = int(float(col[0]))
    


    hor_e = df.iloc[row, :]
    ver_e = df.iloc[:, col]
    
    tck = interpolate.splrep(x, hor_e, s=0.0005, k=4) 
    x_new = np.linspace(min(x), max(x), 10000)
    y_fit = interpolate.BSpline(*tck)(x_new)
    peaks, _ = find_peaks(y_fit)
    peaks_h = x_new[peaks]
        
    
    horizontal_peaks.append(peaks_h)
    horizontal_peaks_position.append(x_new[np.where(y_fit == max(y_fit))[0][0]])
    horizontal_peaks_max.append(df.max().max())
    
    results_half = peak_widths(y_fit, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x_new[-1] - x_new[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(x_new[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(x_new[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    
    results_full = peak_widths(y_fit, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x_new[-1] - x_new[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(x_new[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(x_new[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    
    horizontal_half.append(max(results_half_plot[0]))      
    horizontal_full.append(max(results_full_plot[0]))

    tck = interpolate.splrep(y, ver_e, s=0.0005, k=4) 
    x_new = np.linspace(min(y), max(y), 10000)
    y_fit = interpolate.BSpline(*tck)(x_new)
    peaks, _ = find_peaks(y_fit)
    peaks_v = x_new[peaks]
        
    
    verticle_peaks.append(peaks_v)
    verticle_peaks_position.append(x_new[np.where(y_fit == max(y_fit))[0][0]])
    verticle_peaks_max.append(df.max().max())
    
    results_half = peak_widths(y_fit, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x_new[-1] - x_new[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(x_new[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(x_new[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    
    results_full = peak_widths(y_fit, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x_new[-1] - x_new[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(x_new[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(x_new[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    
    verticle_half.append(max(results_half_plot[0]))      
    verticle_full.append(max(results_full_plot[0]))

plt.plot(z, verticle_full)
plt.plot(z, horizontal_full)
plt.show()



from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors

arr_3d_shortened = arr_3d_loaded[:, :, 80:]
arr_smaller = arr_3d_shortened[::20, ::20, ::20]
max_color = np.max(arr_smaller, axis=None)
min_color = np.min(arr_smaller, axis=None)

axis_0 = arr_smaller.shape[0]
axis_1 = arr_smaller.shape[1]
axis_2 = arr_smaller.shape[2]
arr_empty = np.empty((axis_0, axis_1, 0))



for i in range(axis_2):
    print(i)
    arr_2d = arr_smaller[:, :, i]
    max_val = arr_2d.max(axis=None)
    # Create a boolean mask for data points less than 0.02
    mask = arr_2d < 0.2*max_val
    
    # Set masked values to NaN
    arr_2d[mask] = np.nan
    
    arr_empty = np.concatenate((arr_empty, np.expand_dims(arr_2d, axis=2)), axis=2)



angle = 30, 60, 90, 120, 150, 180, 210, 240, 270, 300

axis_0 = arr_empty.shape[0]
axis_1 = arr_empty.shape[1]
axis_2 = arr_empty.shape[2]

# Create a figure and axis
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
# Create x, y, and z indices
x = np.linspace(-20, 80, num=axis_0)
y = np.linspace(-25, 25, num = axis_1)
z = np.linspace(-5, 45, num = axis_2)


ax.view_init(elev=30, azim=210)
# Create a meshgrid of the indices
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

c = arr_empty.ravel()




# Plot the data as a 3D surface
cp = ax.scatter(xx, yy, zz, c=c, cmap='jet', norm=colors.LogNorm(), alpha=0.5)


ticks = (np.linspace(min_color, max_color, num = 6)).tolist()

# Add labels to the axis
clb=fig.colorbar(cp)

# Custom set the colorbar ticks
ticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.09]
tick_labels = [str(i) for i in ticks]
clb.set_ticks(ticks)
clb.set_ticklabels(tick_labels)

ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=13)
ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=15)
ax.set_zlabel('z-position (µm)', fontsize=18, fontweight="bold", labelpad=15)

ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.zaxis.label.set_fontsize(18)
ax.zaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_zticklabels(ax.get_zticks(), weight='bold')
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

clb.ax.set_title('Electric field (eV)', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)

# Show the plot
plt.show()

