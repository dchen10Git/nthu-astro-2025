import numpy as np
import astropy.units as u
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
from astropy.io import fits
from helpers import get_uJy_cube, get_uJy_per_um_cube, cdelt3, v_to_wavelen
from moment0 import moment0

def specutils_linear_fit(fits_file, center, width, slice=False):
    """
    Fit a linear continuum in each spaxel of a SpectralCube using specutils.
    """
    cube = get_uJy_cube(fits_file)
    delta = (width / 2)
    wavelengths = cube.spectral_axis
    nz, ny, nx = cube.shape
    
    subcube = cube.spectral_slab(center-delta, center+delta)
    sub_wavelengths = subcube.spectral_axis

    # Prepare output cube
    continuum_cube = np.full((nz, ny, nx), np.nan) * u.uJy

    # Loop through each spaxel
    for y in range(ny):
        for x in range(nx):
            flux = subcube[:, y, x]

            # need at least 3 valid points for model
            if np.sum(np.isfinite(flux)) < 3:
                continue  

            try:
                spectrum = Spectrum1D(flux=flux, spectral_axis=sub_wavelengths) # sub_wavelengths so fit stays in window
                # Changing the median window allows for better skipping of peak
                model = fit_generic_continuum(spectrum, 11, model=models.Linear1D()) 
                
                fitted_flux = model(wavelengths)
                continuum_cube[:, y, x] = fitted_flux
            except Exception:
                continue
    
    cont_spec_cube = SpectralCube(data=continuum_cube, wcs=cube.wcs) 
    if slice:
        i_closest = np.argmin(np.abs(wavelengths - center))
        return cont_spec_cube[i_closest,:,:]
        
    return cont_spec_cube
    
def plot_linear_fit(fits_file, center, moment0_width, fit_width):
    lambda_obs = v_to_wavelen(center, -29.8 * u.km/u.s)
    print(f"Lambda obs: {lambda_obs:.5f}")
    spec_slice = specutils_linear_fit(fits_file, lambda_obs, fit_width, True) * moment0_width / cdelt3(fits_file)
    
    plt.imshow(np.log10(spec_slice.value), origin='lower', cmap='inferno',vmin=0,vmax=4.5)
    plt.colorbar(label=f'$\log10 F_\lambda \, ({spec_slice.unit})$')
    plt.title(f'Linear Continuum Fit ($\lambda_0$={center})')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.tight_layout()
    plt.show()
    
# plot_linear_fit(fits_file = 'fits/1s3d.fits', center = 1.644*u.um, moment0_width = 0.003*u.um, fit_width=0.003*u.um)     

def full_spec_continuum(fits_file, window_width=0.1*u.um):
    """
    Compute a full-spectrum continuum cube by fitting linear models in small wavelength windows.
    """
    cube = get_uJy_cube(fits_file)
    wavelengths = cube.spectral_axis
    nz, ny, nx = cube.shape

    # Output arrays
    continuum_sum = np.zeros((nz, ny, nx)) * u.uJy

    start = wavelengths[0]
    end = wavelengths[-1]

    current = start

    while current <= end:
        if current + window_width > end:
            current = end - window_width # so that last fit is not broken
        center = current + window_width / 2
        print(f"Fitting window from {current:.5f} to {(current + window_width):.5f}")

        try:
            partial_cube = specutils_linear_fit(fits_file, center, width=window_width)
            slab = partial_cube.spectral_slab(center - window_width/2, center + window_width/2)
    
            # Find the spectral indices in the full cube that correspond to the partial_cube
            slab_indices = np.where(np.isin(cube.spectral_axis, slab.spectral_axis))[0] 
            for i_slab, i_full in enumerate(slab_indices):
                continuum_sum[i_full, :, :] = slab[i_slab, :, :]
            
        except Exception as e:
            print(f"Window at {center:.5f} failed: {e}")
            pass
        
        if current + window_width < end:
            current += window_width 
        else:
            break

    continuum_cube = SpectralCube(data=continuum_sum, wcs=cube.wcs)

    return continuum_cube

def subtracted_cube(fits_file, center, fit_width=0.005*u.um):
    cube = get_uJy_cube(fits_file)
    cont_cube = specutils_linear_fit(fits_file, center, fit_width)
    return cube - cont_cube

def normalized_cube(fits_file, center, fit_width=0.005*u.um):
    flux_cube = get_uJy_cube(fits_file)
    cont_cube = specutils_linear_fit(fits_file, center, fit_width)
    new_cube = (flux_cube - cont_cube) / cont_cube
    return new_cube

def write_cube(fits_file, center, fit_width=0.005*u.um, name='fits/normal_cube.fits'):
    cube = normalized_cube(fits_file, center, fit_width)
    cube.write(name, format='fits', overwrite=True)
    print("Successfully saved")

fits_file = 'fits/6s3d.fits'
cube = get_uJy_cube(fits_file)
cont_cube = full_spec_continuum(fits_file, 0.06*u.um)
sub_cube = cube - cont_cube
# norm_cube = (cube - cont_cube) / cont_cube
sub_cube.write('fits/subtracted_cube_full6.fits', format='fits', overwrite=True)
print("Successfully saved")