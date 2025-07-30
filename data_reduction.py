from spectral_cube import SpectralCube
import astropy.units as u
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.wcs import WCS
from helpers import cdelt3, v_to_wavelen, get_uJy_cube
from moment0 import moment0
from specutils_fit import specutils_linear_fit, subtracted_cube
from interpolate import interpolate_pixels
from manual_mask import bad_spaxel_mask
from snr import snr_mask

def data_reduce(fits_file, center, moment0_width, fit_width, lambda0=None):
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

def plot_data_reduce(fits_file, center, moment0_width, fit_width):
    lambda_obs = v_to_wavelen(center, (-77)* u.km/u.s) 
    lambda_obs = center
    print(f"Lambda_obs: {lambda_obs:.5f}")
    # slice = data_reduce(fits_file, lambda_obs, moment0_width, fit_width, center)
    slice = fits.open("../fits/slice2d5.fits")[0].data
    
    ax = plt.subplot(111, facecolor='black')
    im = ax.imshow(np.log10(slice), origin='lower', cmap='inferno', vmin=0, vmax=4)
    
    ax.set_title(f'Flux Density ($\lambda_0$={center})', fontsize=12)
    ax.set_xlabel('X pixel', fontsize=12)
    ax.set_ylabel('Y pixel', fontsize=12)
    
    cb = plt.colorbar(mappable=im)
    cb.set_label('$\log_{10} F_\lambda \, (\mu Jy)$', fontsize=12)
    
    # Peak brightness locations
    centers_x = [25.26613227,26.13168151, 27.0120717]
    centers_y = [33.59679364, 38.79008904, 44.07243022]
    
    # H2 Knots
    centers_x = [35, 21, 24]
    centers_y = [41, 46, 35]
    colors = ['blue', 'teal', 'palevioletred']
    
    ax.scatter(centers_x, centers_y, c=colors, marker='x', linewidths=3, edgecolors='black') # edgecolor not really working
    
    plt.tight_layout()
    plt.show()

# plot_data_reduce(fits_file = 'fits/5s3d.fits', center = 2.8025*u.um, moment0_width=0.0015*u.um, fit_width=0.005*u.um)

def write_slice(fits_file, center, moment0_width, fit_width, name='../fits/slice2d.fits'):
    cube = get_uJy_cube(fits_file)
    slice = data_reduce(fits_file, center, moment0_width, fit_width, center)
    
    # Get the 2D WCS (drops spectral axis)
    wcs2d = cube.wcs.sub(['longitude', 'latitude'])

    # Create a FITS header from WCS
    header = wcs2d.to_header()
    
    # Create HDU and save
    hdu = fits.PrimaryHDU(data=slice.value, header=header)
    hdu.writeto(name, overwrite=True)
    print(f"Reduced slice written to file as {name}")

# write_slice(fits_file = 'fits/5s3d.fits', center=2.8025*u.um, moment0_width=0.0015*u.um, fit_width=0.005*u.um, name='fits/slice2d5.fits')

def plot_all(fits_file, center, moment0_width, fit_width):
    # lambda_obs = v_to_wavelen(center, (-29.802)* u.km/u.s) 
    lambda_obs = v_to_wavelen(center, (-77)* u.km/u.s) # systematic v
    # lambda_obs = center 
    print(f"Lambda_obs: {lambda_obs:.5f}")
    fig = plt.figure(figsize=(15, 4))

    # Create GridSpec: 1 row, 6 columns (moment0, linear fit, subtract, masks, interpolate, colorbar)
    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.1], wspace=0.1)

    # Subplots for the 3 maps
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])
    
    # Colorbar axis
    cax = fig.add_subplot(gs[5])
    
    # Plot moment 0 map
    moment0_slice = moment0(fits_file, lambda_obs, moment0_width)
    im0 = ax0.imshow(np.log10(moment0_slice.value), origin='lower', cmap='inferno', vmin=0, vmax=3)
    ax0.set_title(f'Moment 0 Map')
    
    
    # Plot the linear fit map
    cont_slice = specutils_linear_fit(fits_file, lambda_obs, fit_width, True) * moment0_width / cdelt3(fits_file)
    im1 = ax1.imshow(np.log10(cont_slice.value), origin='lower', cmap='inferno',vmin=0,vmax=3)
    ax1.set_title(f'Linear Continuum Fit')
    
    # Plot the subtracted map
    slice = moment0_slice - cont_slice
    
    slice[slice <= 0] = 1e-32 * u.uJy
    im2 = ax2.imshow(np.log10(slice.value), origin='lower', cmap='inferno', vmin = 0,vmax=3)
    ax2.set_title(f'Subtracted Residual')
    
    # Plot the SNR & manually masked map
    # SNR Mask
    slice = snr_mask(fits_file, slice, lambda_obs)
    # Manual bad spaxel masking
    slice = bad_spaxel_mask(center, slice)
    im3 = ax3.imshow(np.log10(slice.value), origin='lower', cmap='inferno', vmin = 0,vmax=3)
    ax3.set_title(f'SNR Masked', fontsize=10)
    
    # Interpolate marked spaxels
    slice = interpolate_pixels(slice)
    im4 = ax4.imshow(np.log10(slice.value), origin='lower', cmap='inferno', vmin = 0,vmax=3)
    ax4.set_title(f'Interpolated')
    
    # Label axes
    ax0.set_ylabel('Y pixel')
    for ax in [ax0,ax1,ax2,ax3,ax4]:
        ax.set_xlabel('X pixel')
        ax.set_xticks(np.arange(0, 60, 10)) 
        ax.set_facecolor('black')
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_yticks([]) # removes extra tick labels
    
    # Add colorbar in its own axis
    fig.colorbar(im0, cax=cax, label=r'$\log_{10} F_\lambda \, (\mu Jy)$')
    fig.suptitle(f"$\lambda_0$={center}", fontsize=14)
    fig.subplots_adjust(top=0.88, bottom=0.3)
    plt.tight_layout()
    plt.show()

# plot_all(fits_file = 'fits/4s3d.fits', center = 1.644*u.um, moment0_width=0.0015*u.um, fit_width=0.005*u.um)

def compare_interpolation(fits_file, center, width):
    fig = plt.figure(figsize=(10, 5))

    # Create GridSpec: 1 row, 4 columns (last for colorbar)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.1], wspace=0.2)

    # Subplots for the 3 maps
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    
    # Colorbar axis
    cax = fig.add_subplot(gs[2])
    
    # Plot the residual map
    slice_2d = data_reduce(fits_file, center, width, cdelt3(fits_file).value)
    im0 = ax0.imshow(np.log10(slice_2d.value), origin='lower', cmap='inferno', vmin = 0,vmax=3)
    ax0.set_title(f'Residual Flux Density ({center} μm)')
    
    # Plot the interpolated residual map
    interp_slice_2d = interpolate_pixels(slice_2d.value)
    im1 = ax1.imshow(np.log10(interp_slice_2d), origin='lower', cmap='inferno', vmin = 0,vmax=3)
    ax1.set_title(f'Interpolated Residual ({center} μm)')
    
    # Label axes
    ax0.set_ylabel('Y pixel')
    for ax in [ax0,ax1]:
        ax.set_xlabel('X pixel')
    
    # Add colorbar in its own axis, matching last plot's mappable (im2)
    fig.colorbar(im1, cax=cax, label=r'$\log_{10} F_\lambda \, (μJy)$')
    
    plt.tight_layout()
    plt.show()
    
# compare_interpolation(fits_file = 'fits/4s3d.fits', center = 1.644, width = 0.001)