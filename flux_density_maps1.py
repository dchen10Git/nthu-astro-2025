import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.constants import c
from astropy.utils import data
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

from helpers import filter_bad, get_uJy_cube

hdulist = fits.open('../fits/1s3d.fits')  
cube = get_uJy_cube(hdulist)

# Rest wavelength ([Fe II] 1.644 µm)
lambda0 = 1.644 * u.micron

# Get wavelength axis
wavelengths = cube.spectral_axis.to(u.micron)

moment0 = cube.moment(order=0)  # integrated intensity


wcs_2d = cube.wcs.sub(['longitude', 'latitude'])

fig, axes = plt.subplots(5, 5, figsize=(10, 8), subplot_kw={'projection': wcs_2d})

indices = list(range(0,len(cube))) 

bad_indices = filter_bad(hdulist)

filtered_indices = [item for item in indices if item not in bad_indices]

b = list(range(0, len(filtered_indices),round(len(filtered_indices)/25)))

indices = [filtered_indices[i] for i in b]

im = None
for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
    slice_i = cube[idx, :, :]
    data = slice_i.hdu.data
    data[data <= 0] = np.nan  # mask out zeros and negatives to avoid log10 issues
    log_data = np.log10(data)
    ax.set_facecolor('black')
    im = ax.imshow(log_data, origin='lower', cmap='inferno')
    
    # Add channel label
    ax.text(0.05, 0.95, round(wavelengths[idx].to(u.um), 3), transform=ax.transAxes,
            fontsize=5, color='white', va='top', ha='left')
    
    # Turn off axis labels and ticks initially
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Show Dec ticks and label only on left column
    if i % 5 == 0:  # first column plots (indices 0 and 3)
        ax.coords[1].set_axislabel('Dec (")')
        ax.coords[1].set_ticklabel_visible(True)
        ax.coords[1].set_ticks_visible(True)
    else:
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_ticks_visible(False)
    
    # Show RA ticks and label only on bottom row
    if i // 5 == 4:  # second row plots (indices 3,4,5)
        ax.coords[0].set_axislabel('RA (")')
        ax.coords[0].set_ticklabel_visible(True)
        ax.coords[0].set_ticks_visible(True)
    else:
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[0].set_ticks_visible(False)
        
    ax.coords[0].set_ticklabel(exclude_overlapping=True)

            

plt.tight_layout()
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                    fraction=0.05, pad=0.1)
cbar.set_label(r"$\log_{10} F_\lambda \, (\mu Jy)$")

plt.show()
