import numpy as np
from astropy.io import fits
import astropy.units as u
from spectral_cube import SpectralCube

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
