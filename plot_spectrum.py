import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy import constants as const 
from helpers import get_error_spectrum, v_to_wavelen
from spectral_cube import SpectralCube
from astropy.table import Table
from line_detection import detect_lines, detect_lines_from_table
import re

def spec_at_pixel(cube, center, width, x, y):
    '''
    Takes a spectral cube (units uJy) and plots based on wavelength
    at a given center and width, at a given x, y spaxel.
    Also plots error bars if error cube is given.
    '''
    delta = (width / 2)
    subcube = cube.spectral_slab(center-delta, center+delta)
    flux_spec = subcube[:, y, x]  # shape: (n_channels,)
    vcube = subcube.with_spectral_unit(u.km/u.s, velocity_convention='optical',rest_value=center)
    v = vcube.spectral_axis
    
    return v, flux_spec


def plot_spec_at_pixel(fits_file, subtracted_cube, center, width, pixel_list, velocity=False, lines=None):
    cube = subtracted_cube
    plt.figure(figsize=(12, 6))

    colors = ['blue', 'teal', 'palevioletred']
    for i, (x, y) in enumerate(pixel_list):
        v, flux = spec_at_pixel(cube, center, width, x, y)
        wavelengths = v_to_wavelen(center, v) # for plotting using wavelength
        error = get_error_spectrum(fits_file, subtracted_cube, x, y)
        label = f"Pixel ({x}, {y})" 
        
        if velocity:
            plt.plot(v, flux, drawstyle='steps-mid', color=colors[i], label=label)
        else:
            plt.plot(wavelengths, flux, drawstyle='steps-mid', color=colors[i], label=label)
        # Error bars
        if error is not None:
            if velocity:
                plt.fill_between(v, (flux - error).value, (flux + error).value, step='mid', alpha=0.4, color=colors[i])
            else:
                plt.fill_between(wavelengths, (flux - error).value, (flux + error).value, step='mid', alpha=0.4, color=colors[i])
    
    if velocity:    
        plt.axvline(0, color='black', linestyle='-',alpha=0.7)
        plt.axhline(0, color='blue', linestyle='--', alpha=0.4)
        plt.xlabel('Velocity (km/s)', fontsize=16)
    else: 
        plt.xlabel('Wavelength (μm)', fontsize=16)
        if lines is not None:
            for wav, flux, _ in lines:
                plt.plot(wav.value, flux.value * 1.3, color='r', marker='|', markersize=6)
                # plt.text(wav.value, 15, f"{wav:.3f}", rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=8, color='gray')

    plt.ylabel('Flux (μJy)', fontsize=16)
    # plt.ylim(-3, 20)
    plt.title(f'Emission Line Profile', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spec(wavelengths, flux, error=None, x=None, y=None, lines=None, database_lines=None):
    '''
    Plots spectrum given arrays of wavelength, flux, and error (optional)
    Can indicate given lines and color them based on lines_dict
    '''       
    
    fig, ax = plt.subplots(figsize=(12, 6))
    label = f"Pixel ({x}, {y})" 

    ax.plot(wavelengths, flux, drawstyle='steps-mid', color='blue', label=label)
    # Error bars
    if error is not None:
        ax.fill_between(wavelengths, (flux - error).value, (flux + error).value, step='mid', alpha=0.4, color='blue')
    
        if lines is not None and database_lines is not None:
            # Gets all rows containing H2
            H2_table = database_lines[['H2' in s for s in database_lines['species_list']]]
            H2_wavelengths = H2_table['rest_wavelength']
            
            # Gets all rows containing [Fe II]
            FeII_table = database_lines[['[Fe II]' in s for s in database_lines['species_list']]]
            FeII_wavelengths = FeII_table['rest_wavelength']
            
            # Gets all rows containing H I
            HI_table = database_lines[['H I' in s for s in database_lines['species_list']]]
            HI_wavelengths = HI_table['rest_wavelength']
            
            # Gets all rows containing He I (note this is marked as He in the lines table)
            HeI_table = database_lines[['He' in s for s in database_lines['species_list']]]
            HeI_wavelengths = HeI_table['rest_wavelength']
            
            for wav, fluxes, snr in lines:
                H2_mask = np.abs(H2_wavelengths - wav) <= 0.001
                FeII_mask = np.abs(FeII_wavelengths - wav) <= 0.001
                HI_mask = np.abs(HI_wavelengths - wav) <= 0.001
                HeI_mask = np.abs(HeI_wavelengths - wav) <= 0.001
                
                if np.any(H2_mask):
                    transition_label = H2_table['transition_labels'][H2_mask]
                    label_str = str(*transition_label)
                    match = re.search(r"(H2[^',]*)", label_str) # Regex for displaying only H2 line transition info
                    ax.plot(wav, fluxes + fluxes / snr + 1, color='r', marker='v', markersize=4)
                    ax.text(wav, fluxes + fluxes / snr + 2, f"{wav:.3f}, " + match.group(1), rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=8, color='black')

                    # Search for 0-0 transitions for rotational diagram
                    match = re.search(r"v=0-0", label_str)
                    if match:
                        print(fluxes, label_str)
                    
                    
                elif np.any(FeII_mask):
                    transition_label = FeII_table['transition_labels'][FeII_mask]
                    label_str = str(*transition_label)
                    match = re.search(r"(\[Fe II\][^,]*)", label_str) # Regex for displaying only [Fe II] line transition info
                    ax.plot(wav, fluxes + fluxes / snr + 1, color='green', marker='v', markersize=4)
                    ax.text(wav, fluxes + fluxes / snr + 2, f"{wav:.3f}, " + match.group(1), rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=8, color='black')
                
                elif np.any(HI_mask):
                    transition_label = HI_table['transition_labels'][HI_mask]
                    label_str = str(*transition_label)
                    match = re.search(r"(H I[^',]*)", label_str) # Regex for displaying only H I line transition info
                    ax.plot(wav, fluxes + fluxes / snr + 1, color='orange', marker='v', markersize=4)
                    ax.text(wav, fluxes + fluxes / snr + 2, f"{wav:.3f}, " + match.group(1), rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=8, color='black')
                
                elif np.any(HeI_mask):
                    transition_label = HeI_table['transition_labels'][HeI_mask]
                    label_str = str(*transition_label)
                    match = re.search(r"(He[^',]*)", label_str) # Regex for displaying only He I line transition info
                    ax.plot(wav, fluxes + fluxes / snr + 1, color='purple', marker='v', markersize=4)
                    ax.text(wav, fluxes + fluxes / snr + 2, f"{wav:.3f}, " + match.group(1), rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=8, color='black')

                else:
                    # Plots non-H2 lines
                    ax.plot(wav, fluxes + fluxes / snr + 1, color='black', marker='v', markersize=2)
            
            # Prints database values not detected
            detected = 0
            dupes = 0
            for wav in database_lines['rest_wavelength']:
                if np.any(np.abs(lines['wavelength'] - wav) <= 0.001):
                    # print(f'{wav:.5f} detected')
                    detected += 1
                    # counts duplicates
                    if np.sum(np.abs(lines['wavelength'] - wav) <= 0.001) > 1:
                        dupes += 1
                        # print(f'{wav:.5f} has duplicates')
                else:
                    pass 
                    # print(f'{wav:.5f} not detected')
            print(f'total detected: {detected}')
            print(f'total undetected: {len(database_lines) - detected}')
            print(f'total dupes: {dupes}')
                
    plt.xlabel('Wavelength (μm)', fontsize=16)
    plt.ylabel('Flux (μJy)', fontsize=16)
    # plt.ylim(-3, 20)
    plt.title(f'Emission Line Profile', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
     
'''
Available spaxels:
    (23, 26) - Approx star location
    (25, 34) - Brightest atomic jet knot
    (35, 41) - Brightest H2 lobe
    
[(25.26613227, 33.59679364), (26.13168151, 38.79008904), (27.0120717, 44.07243022)] # peak brightnesses
[(35, 41), (21, 46), (24, 35)] # Molecular ring lobes
'''

pixels = [(25, 34)]
table = Table.read(f'fits/merged_spectrum{pixels[0][0]}_{pixels[0][1]}.fits')
detected_lines = Table.read(f'fits/detected_lines{pixels[0][0]}_{pixels[0][1]}.fits')
database_table = Table.read('fits/grouped_lines_summary.fits')

plot_spec(table['wavelength'], table['flux'], table['error'], pixels[0][0], pixels[0][1], detected_lines, database_table)

'''
=== Code for plotting spectrum of entire fits file ===
fits_file = 'fits/4s3d.fits'
subtracted_cube = SpectralCube.read('fits/subtracted_cube_full4.fits')

for lam, flux, snr_val in lines:
    print(f"Wavelength: {lam:.4f}, SNR: {snr_val:.2f}")

print(f"Line count: {len(lines)}")

if fits_file == 'fits/4s3d.fits':
    plot_spec_at_pixel(fits_file, subtracted_cube, 1.4*u.um, 1*u.um, pixels, False, lines)
elif fits_file == 'fits/5s3d.fits':
    plot_spec_at_pixel(fits_file, subtracted_cube, 2.42*u.um, 1.6*u.um, pixels, False, lines)
elif fits_file == 'fits/6s3d.fits':
    plot_spec_at_pixel(fits_file, subtracted_cube, 4*u.um, 2.3*u.um, pixels, False, lines)

Use 24, 28 for Fe II; 32, 42 for H2
'''


