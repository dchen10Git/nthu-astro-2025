import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from spectral_cube import SpectralCube
import numpy as np
from astropy.coordinates import SkyCoord

def get_uJy_cube(fits_file):
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    pixar_sr = header['PIXAR_SR'] * u.sr # pixel area in steradians
    cube = SpectralCube.read(fits_file,hdu=1)
    cube = (cube * pixar_sr).to(u.uJy)
    return cube

def get_uJy_per_um_cube(fits_file):
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    cdelt = header['CDELT3'] * u.um # wavelength increment
    cube = get_uJy_cube(fits_file)
    new_cube = cube / cdelt # in uJy/um
    return new_cube

# idk about those two...
def get_uJy_error_cube(fits_file):
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    pixar_sr = header['PIXAR_SR'] # pixel area in steradians
    error_data = hdul['ERR'].data * 1e12
    cube = SpectralCube.read(fits_file,hdu=1).to(u.uJy/u.sr)
    error_cube = SpectralCube(data=error_data,wcs=cube.wcs)
    error_cube = (error_cube * pixar_sr) * u.uJy
    return error_cube

def get_uJy_per_um_error_cube(fits_file):
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    cdelt = header['CDELT3'] * u.um # wavelength increment
    cube = get_uJy_error_cube(fits_file)
    new_cube = cube / cdelt
    return new_cube

def get_error_spectrum(fits_file, subtracted_cube, x, y):
    # Extract the spectrum at the spaxel
    spectrum = subtracted_cube.unmasked_data[:, y, x]  # shape (N_channels,)
    wavelengths = subtracted_cube.spectral_axis.to('um')  # shape (N_channels,)
        
    error_cube = get_uJy_error_cube(fits_file)
    error_spec = (error_cube[:, y, x])
        
    # Compute calibration error (5% of flux)
    calib_error = 0.05 * spectrum

    # Total error
    total_error = np.sqrt(error_spec ** 2 + calib_error ** 2)
    
    return total_error

def velosys(fits_file):
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    return header['VELOSYS'] * u.m/u.s # this is usually negative

def pixar_sr(fits_file):
    '''
    Given a fits file, returns the pixel area in steradians.
    '''
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    return header['PIXAR_SR'] * u.sr # pixel area in steradians

def cdelt1(fits_file):
    '''
    Returns spatial increment in degrees.
    '''
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    return header['CDELT1'] * u.degree

def cdelt3(fits_file):
    '''
    Returns wavelength increment in microns.
    '''
    hdul = fits.open(fits_file)
    header = hdul['SCI'].header
    return header['CDELT3'] * u.um

def target_ra_dec(fits_file):
    hdul = fits.open(fits_file)
    header = hdul['PRIMARY'].header
    return header['TARG_RA'] * u.degree, header['TARG_DEC'] * u.degree

# print(target_ra_dec('fits/1s3d.fits'))

def get_pixel(cube, ra, dec):
    skycoord = SkyCoord(ra=ra, dec=dec, frame='icrs')

    # Get the WCS for the spatial dimensions
    wcs = cube.wcs.sub(['longitude', 'latitude'])

    # Convert to pixel coordinates
    x_pix, y_pix = wcs.world_to_pixel(skycoord)
    return x_pix.item(), y_pix.item()

def get_target_pixel(fits_file):
    cube = SpectralCube.read(fits_file,hdu=1)
    ra, dec = target_ra_dec(fits_file)
    return get_pixel(cube, ra, dec)

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

def v_to_wavelen(lambda_0, v):
    '''
    Gives lambda_obs given rest wavelength and redshift velocity (blueshift = negative).
    '''
    return (lambda_0 * (v.to(u.km/u.s) + const.c)/const.c).to(u.um)

def wavelen_to_v(lambda_0, lambda_obs):
    return (const.c * (lambda_obs - lambda_0) / lambda_0).to(u.km/u.s)
