import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.constants import c
from astropy.utils import data
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from spectral_cube import SpectralCube

# === Filtering Out Bad Data ===

def filter_bad(hdul):
    # Open file and get DQ array
    # hdul = fits.open('fits/2s3d.fits')
    dq = hdul['DQ'].data  # shape: (lambda, y, x)

    # Define which DQ bits are "bad"
    BAD_BITS = 1 + 2 + 8 + 16  # Do not use, Saturated, Cosmic ray, Out of field

    # Mask = True where pixel is bad
    bad_mask = (dq & BAD_BITS) != 0  # shape: (lambda, y, x)

    # Calculate fraction of bad pixels per slice
    bad_fraction_per_slice = bad_mask.reshape(bad_mask.shape[0], -1).mean(axis=1)

    # Define threshold for "bad" slices
    threshold = 0.5  # change this if necessary

    # Get indices of slices that are mostly bad
    bad_slice_indices = np.where(bad_fraction_per_slice > threshold)[0]

    #print("Bad slice indices (>", threshold*100, "% bad):")
    return bad_slice_indices
    #print(len(bad_slice_indices))
    #print(len(dq))