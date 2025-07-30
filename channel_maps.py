import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube
import astropy.constants as const
from data_reduction import data_reduce
from helpers import get_pixel, get_uJy_cube

def generate_slices(fits_file, lambda_obs, v_centers, v_width=40*u.km/u.s, fit_width=0.005 * u.um, lambda_0=None):
    """
    Given a cube and velocity bin edges, return a list of 2D slices collapsed over each velocity bin.
    """
    cube = SpectralCube.read(fits_file, hdu=1)
    slices = []
    delta = v_width / 2
    
    for i in range(len(v_centers)):
        v_lo = v_centers[i] - delta
        v_hi = v_centers[i] + delta

        lambda_lo = lambda_obs * (1 + v_lo / const.c)
        lambda_hi = lambda_obs * (1 + v_hi / const.c)
        center_lambda = (lambda_lo + lambda_hi) / 2
        width = lambda_hi - lambda_lo

        slice_2d = data_reduce(fits_file, center_lambda, width, fit_width, lambda_0)
        slices.append(slice_2d)

        # Get the 2D WCS (drops spectral axis)
        wcs2d = cube.wcs.sub(['longitude', 'latitude'])

        # Create a FITS header from WCS
        header = wcs2d.to_header()
        
        # Create HDU and save
        hdu = fits.PrimaryHDU(data=slice_2d.value, header=header)
        name = f"fits/channel_map5{i+1}.fits"
        hdu.writeto(name, overwrite=True)
        print(f"{i+1}/{len(v_centers)} Reduced slice written to file as {name}")

# For 1.644
v_centers = [-154, -111, -68, -25, 17, 146, 189, 232] * u.km/u.s # spaced out based on CDELT3 increment size

# For 2.803
v_centers = [-110, -68, -25, 17, 59] * u.km/u.s
# generate_slices('fits/5s3d.fits', 2.8025*u.um, v_centers)

def plot_channel_maps(slices, v_centers):
    """
    Plot the 2D channel slices with titles showing center velocities and a shared colorbar.
    """
    ra = 69.896675 * u.degree
    dec = 25.69561666667 * u.degree
    x1, y1 = get_pixel(get_uJy_cube('fits/5s3d.fits'), ra, dec)
    x1 -= 2
    y1 -= 1 # THESE ARE NOT REAL
        
    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(1, 6, figure=fig, width_ratios=[1,1,1,1,1,0.1])
    axes = [fig.add_subplot(gs[i]) for i in range(5)] # + [fig.add_subplot(gs[i]) for i in range(5,9)]

    for i, (slice_2d, v_center) in enumerate(zip(slices, v_centers)):
        im = axes[i].imshow(np.log10(slice_2d), origin='lower', cmap='inferno', vmin=0, vmax=2.5)
        axes[i].text(0.02, 0.95, f"{v_center:.0f}", color='white', fontsize=11, transform=axes[i].transAxes, ha='left', va='top')
        axes[i].plot(x1, y1, marker='*', color='gold')
        axes[i].set_facecolor('black')
        
        # Overlay contours (in linear space)
        # 1.644
        contour_levels = [1.5, 2.5, 5, 15]  # in μJy
        
        # 2.803
        contour_levels = [3,5,10,30]
        
        axes[i].contour(slice_2d, levels=contour_levels, colors='cyan', linewidths=0.6)

        if i in [0,4]:
            axes[i].set_ylabel('Y pixel')
        else:
            axes[i].set_ylabel('')
            axes[i].set_yticks([])
        if i in [4,5,6,7,8]:
            axes[i].set_xlabel('X pixel')
        else:
            axes[i].set_xlabel('')
        
        # temp
        axes[i].set_xlabel('X pixel')
        axes[i].set_xticks([0,10,20,30,40,50])
        axes[4].set_ylabel('')
        axes[4].set_yticks([])
        
    # Add colorbar
    # cbar_ax = fig.add_subplot(gs[:, 4])
    cbar_ax = fig.add_subplot(gs[5])
    fig.colorbar(im, cax=cbar_ax, label='$\log_{10} F_\lambda \, (\mu Jy)$')
    fig.subplots_adjust(top=0.7, bottom=0.3)
    # plt.tight_layout()
    plt.show()

# Make sure to change number of maps shown
slices = [fits.open(f"fits/channel_map5{i+1}.fits")[0].data for i in range(5)]
plot_channel_maps(slices, v_centers)