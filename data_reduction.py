from spectral_cube import SpectralCube
import astropy.units as u
from astropy.io import fits
import numpy as np
from helpers import cdelt3
from moment0 import moment0
from specutils_fit import specutils_linear_fit
from scipy.interpolate import griddata

def snr_mask(fits_file, slice, center, threshold=3):
    '''
    Takes a slice, and based on ERR data, masks all spaxels with SNR < threshold to 1e-32.
    '''
    # Load cube and error cube, convert error cube to uJy
    hdul = fits.open(fits_file)
    cube = SpectralCube.read(fits_file,hdu=1)
    flux_array = hdul['SCI'].data
    error_array = hdul['ERR'].data
    
    wavelengths = cube.spectral_axis
    i_closest = np.argmin(np.abs(wavelengths - center))
    flux_slice = flux_array[i_closest,:,:]
    error_slice = error_array[i_closest,:,:]
    
    # Compute calibration error (5% of flux)
    calib_error = 0.05 * flux_slice
    
    # Total error
    sigma_total = np.sqrt(error_slice**2 + calib_error**2)

    # Compute SNR
    snr = (flux_slice / sigma_total) # flux_slice or slice.value?
    
    # Debugging
    # x,y=34,9
    # print(f"x,y={29},{28}")
    # print(f"Calibration error: {calib_error[y,x]}")
    # print(f"ERR: {error_slice[y,x]}")
    # print(f"Total error: {sigma_total[y,x]}")
    # print(f"Flux: {slice[y,x]}")
    # print(f"SNR: {snr[y,x]}")

    # Create output where SNR < 3 becomes 1e-32, others keep original
    masked_slice = np.where(snr < threshold, 1e-32 * u.uJy, slice)

    return masked_slice

def bad_spaxel_mask(lambda_0, slice_2d):
    """
    The following function can be used to add a list of coordinates 
    of "bad" coordinates to be masked out (i.e. changed to 1e-32).
    """
    bad_coords = []
    # [Fe II]
    if lambda_0 == 1.644 * u.um:
        bad_coords = [(17,30),(17,31),(17,35),(17,36),(18,30),(18,31),(18,33),(18,34),(18,35),(18,36),(19,30),(19,33),(20,30),(20,31),(20,33),(20,34),(21,33),(21,34),
                        (35,40),(36,40),(35,39),(36,39),(38,39),(39,39),(36,38),(37,38),(38,38),(39,38),(36,37),(37,37),(38,37),(39,37),(35,35),(36,35),(37,35),(38,35),(36,34),(37,34),(38,34)]
                        # (13,22),(14,22),(15,22),(16,22),(13,23),(14,23),(15,23),(16,23),(13,26),(14,26),(15,26),(16,26),(14,25),(13,27),(16,27),(12,28),(13,28),(14,28),(13,24)]
    elif lambda_0 == 2.803 * u.um:
        bad_coords = [(18,22),(19,22),(18,21),(19,21),(19,20),(20,20),(21,20),(22,20),(19,19),(20,19),(21,19),(18,17),(19,17),(20,17),(21,17),(18,16),(19,16),(20,16),(21,16)]
    
    for (x, y) in bad_coords:
        slice_2d[y, x] = 1e-32 * u.uJy # value to denote pixel to be interpolated
    
    return slice_2d

def interpolate_pixels(slice, target_value=1e-32, method='linear', fallback='nearest'):
    """
    Interpolates pixels in a 2D array (units uJy) that are equal to a specific target value.
    """
    flux_array = slice.value
    ny, nx = slice.shape
    yy, xx = np.indices((ny, nx))

    # Get all valid points (not NaN and not equal to target_value)
    valid_mask = (~np.isnan(flux_array)) & (flux_array != target_value)
    # print(valid_mask)
    known_points = np.stack([yy[valid_mask], xx[valid_mask]], axis=-1)
    known_values = flux_array[valid_mask]

    # Get all target pixels to interpolate
    target_mask = (flux_array == target_value)
    target_points = np.stack([yy[target_mask], xx[target_mask]], axis=-1)

    # Interpolate one pixel at a time
    for (yi, xi) in target_points:
        try:
            interp_value = griddata(
                known_points,
                known_values,
                np.array([[yi, xi]]),
                method=method
            )[0]
        except Exception:
            interp_value = np.nan

        # Try fallback if primary method failed or returned nan
        if (fallback is not None) and (np.isnan(interp_value)):
            try:
                interp_value = griddata(
                    known_points,
                    known_values,
                    np.array([[yi, xi]]),
                    method=fallback
                )[0]
            except Exception:
                interp_value = target_value  # fallback also failed, keep original

        # Replace the value in the copy
        flux_array[yi, xi] = interp_value

    return flux_array * u.uJy

def data_reduce(fits_file, center, moment0_width, fit_width, lambda0=None):
    """
    1. Get the moment-0 map using moment0()
    2. Compute the linear continuum using specutils_linear_fit()
    3. Compute the subtracted slice 
    4. Performs an SNR mask using snr_mask()
    5. If applicable, masks out manually selected bad spaxels
    6. Interpolates any masked out spaxels.

    """
    moment0_slice = moment0(fits_file, center, moment0_width)
    cont_slice = specutils_linear_fit(fits_file, center, fit_width, True) * moment0_width / cdelt3(fits_file)
    
    # Slice Subtraction
    slice = moment0_slice - cont_slice
    
    # Masks away all nonpositives to avoid log issues
    slice[slice <= 0] = 1e-32 * u.uJy
    
    # SNR Mask
    slice = snr_mask(fits_file, slice, center)
    
    # Manual bad spaxel masking
    if lambda0 is not None:
        slice = bad_spaxel_mask(lambda0, slice)
    
    # Interpolate marked spaxels
    slice = interpolate_pixels(slice)
    
    return slice # in uJy