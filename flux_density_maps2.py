import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.constants import c
from astropy.utils import data
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

fn = data.get_pkg_data_filename('tests/data/example_cube.fits', 'spectral_cube')
cube = SpectralCube.read(fn)
print(cube)

# Rest wavelength ([Fe II] 1.644 µm)
lambda0 = 1.644 * u.micron

# Extract slab in that wavelength range
slab = cube.spectral_slab(-150 * u.km/u.s, 150 * u.km/u.s)

moment0 = slab.moment(order=0)  # integrated intensity

data = moment0.hdu.data  # 2D numpy array

wcs_2d = cube.wcs.sub(['longitude', 'latitude'])

fig, axes = plt.subplots(2, 3, figsize=(10, 8), subplot_kw={'projection': wcs_2d})

indices = [1, 2, 3, 4, 5, 6]

im = None
for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
    slice_i = cube[idx, :, :]
    im = ax.imshow(slice_i.hdu.data, origin='lower', cmap='inferno')
    
    # Add channel label
    ax.text(0.05, 0.95, f"Channel {idx}", transform=ax.transAxes,
            fontsize=10, color='white', va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.5))
    
    # Turn off axis labels and ticks initially
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Show Dec ticks and label only on left column
    if i % 3 == 0:  # first column plots (indices 0 and 3)
        ax.coords[1].set_axislabel('Dec (")')
        ax.coords[1].set_ticklabel_visible(True)
        ax.coords[1].set_ticks_visible(True)
    else:
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_ticks_visible(False)
    
    # Show RA ticks and label only on bottom row
    if i // 3 == 1:  # second row plots (indices 3,4,5)
        ax.coords[0].set_axislabel('RA (")')
        ax.coords[0].set_ticklabel_visible(True)
        ax.coords[0].set_ticks_visible(True)
    else:
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[0].set_ticks_visible(False)
        
    ax.coords[0].set_ticklabel(exclude_overlapping=True)

            

plt.tight_layout(rect=[0, 0.05, 1, 1])
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                    fraction=0.06, pad=0.1)
cbar.set_label(r"$\log_{10} F_\lambda (\mu Jy)$")

plt.show()
