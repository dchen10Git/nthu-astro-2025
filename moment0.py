import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from helpers import get_uJy_per_um_cube, v_to_wavelen

def moment0(fits_file, center, width, cube=None):
    '''
    Takes either a fits file, or spectral cube (units uJy/um), 
    and creates a 2D moment 0 map at a given center and width.
    '''
    # Gets spectral cube if not provided
    if cube is None:
        cube = get_uJy_per_um_cube(fits_file)
    
    assert cube.unit == u.uJy/u.um
    
    delta = (width / 2)
    
    # Extract subcube in the window
    subcube = cube.spectral_slab(center - delta, center + delta)

    # Calculate moment 0 (sum along spectral axis) in flux units
    moment0_map = subcube.moment(order=0)  # This integrates flux along spectral axis
    return moment0_map # this is a 2D map with values

def plot_moment0(fits_file, center, width):
    lambda_obs = v_to_wavelen(center, -(75-29.8) * u.km/u.s)
    print(f"Lambda obs: {lambda_obs:.5f}")
    moment0_map = moment0(fits_file, lambda_obs, width)
    
    # from snr import snr_mask
    # moment0_map = snr_mask(fits_file, moment0_map, lambda_obs, 3)

    # Plot moment 0 map
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(moment0_map.value), origin='lower', cmap='inferno', vmin=0, vmax=4.5)
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    ax.set_title(f'Moment 0 Map ($\lambda_0$={center})')
    fig.colorbar(im, ax=ax, label='$\log_{10} F_\lambda \, (μJy)$')
    plt.tight_layout()
    plt.show()
    
# plot_moment0(fits_file = 'fits/1s3d.fits', center = 1.644 * u.um, width = 0.003 * u.um)

