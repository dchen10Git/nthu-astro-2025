import numpy as np
from spectral_cube import SpectralCube
from scipy.signal import find_peaks
import astropy.units as u
from astropy.table import Table
from helpers import get_error_spectrum

def detect_lines(fits_file, subtracted_cube, x, y, snr_thresh=3):
    spectrum = subtracted_cube.unmasked_data[:, y, x]  # shape (N_channels,)
    wavelengths = subtracted_cube.spectral_axis.to('um')  # shape (N_channels,)
    total_error = get_error_spectrum(fits_file, subtracted_cube, x, y)

    snr = spectrum / total_error

    # Find peaks in the spectrum with SNR > threshold
    peak_indices, _ = find_peaks(snr, height=snr_thresh)
    
    # Extract peak wavelengths and SNRs
    peak_wavelengths = wavelengths[peak_indices]
    peak_snr_values = snr[peak_indices]
    peak_fluxes = spectrum[peak_indices]
    return list(zip(peak_wavelengths, peak_fluxes, peak_snr_values))

def detect_lines_from_table(wavelengths, flux, error, x, y, snr_thresh=3):
    snr = flux / error

    # Find peaks in the spectrum with SNR > threshold
    peak_indices, _ = find_peaks(snr, height=snr_thresh)
    
    # Extract peak wavelengths and SNRs
    peak_wavelengths = wavelengths[peak_indices]
    peak_fluxes = flux[peak_indices]
    peak_snr_values = snr[peak_indices]
    return list(zip(peak_wavelengths, peak_fluxes, peak_snr_values))

def make_line_table(x, y):
    # print(f'x, y: ({x}, {y})')

    subtracted_cube1 = SpectralCube.read('../fits/subtracted_cube_full4.fits')
    subtracted_cube2 = SpectralCube.read('../fits/subtracted_cube_full5.fits')
    subtracted_cube3 = SpectralCube.read('../fits/subtracted_cube_full6.fits')

    lines1 = detect_lines('../fits/4s3d.fits', subtracted_cube1, x, y, snr_thresh=3)
    lines2 = detect_lines('../fits/5s3d.fits', subtracted_cube2, x, y, snr_thresh=3)
    lines3 = detect_lines('../fits/6s3d.fits', subtracted_cube3, x, y, snr_thresh=3)

    combined_lines = lines1 + lines2 + lines3
    # for lam, flux, snr_val in combined_lines:
    #     print(f"Wavelength: {lam:.4f}, SNR: {snr_val:.2f}")

    # print(f"Line count: {len(combined_lines)}")

    wavelengths, flux, snr = zip(*combined_lines)

    tbl = Table([wavelengths, flux, snr], names=('wavelength', 'flux', 'snr'))
    
    tbl.sort('wavelength')

    wavelengths = tbl['wavelength']
    
    # Temporary list to store rows to keep
    rows_to_keep = [0]  # start by keeping the first row

    for i in range(1, len(wavelengths)):
        # If it's far enough from the previous kept row, keep it
        if abs(wavelengths[i] - wavelengths[rows_to_keep[-1]]) > 0.005:
            rows_to_keep.append(i)
        else:
            # Check which one has higher value (e.g. in 'snr')
            if tbl['flux'][i] > tbl['flux'][rows_to_keep[-1]]:
                rows_to_keep[-1] = i  # replace with the better one

    # Apply the filter
    clean_tbl = tbl[rows_to_keep]

    clean_tbl.write(f'../fits/detected_lines{x}_{y}.fits', format='fits', overwrite=True)
    # print(clean_tbl)
    
    print(f"Successfully saved as fits/detected_lines{x}_{y}.fits")
    
# make_line_table(23,26)
# make_line_table(25,34)
# make_line_table(35,41)

def make_latex_table(x,y):
    table = Table.read(f'../fits/detected_lines{x}_{y}.fits', format="fits")
    # Peek at the first few rows
    print(table[:5])

    # Convert to pandas
    df = table[:20].to_pandas()

    # Export to LaTeX
    latex_table = df.to_latex(f"../Line Tables/lines_table{x}_{y}.tex",
    index=False,  # To not include the DataFrame index as a column in the table
    caption="20 Detected Lines",  # The caption to appear above the table in the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="lccc",  # The format of the columns: left-aligend first column and center-aligned remaining columns as per APA guidelines
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.2f}".format  # Formats floats to two decimal places
    )

    print(latex_table)
    print(f"Saved as lines_table{x}_{y}.tex")
        
# make_latex_table(23,26)
# make_latex_table(25,34)
# make_latex_table(35,41)