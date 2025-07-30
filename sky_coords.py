# # === Extract 2D WCS for celestial coordinates ===
# wcs_2d = cube.wcs.sub(['longitude', 'latitude'])

# # === Get center RA/Dec ===
# ny, nx = slice_2d.shape
# center_x, center_y = nx // 2, ny // 2
# ra_center, dec_center = wcs_2d.all_pix2world(center_x, center_y, 0)  # degrees

# # === Create grids of RA/Dec offset in arcsec ===
# y, x = np.mgrid[:ny, :nx]
# ra, dec = wcs_2d.all_pix2world(x, y, 0)  # degrees
# dra = (ra - ra_center) * 3600 * np.cos(np.deg2rad(dec_center))  # arcsec
# ddec = (dec - dec_center) * 3600  # arcsec

# === Plot ===
#im = plt.pcolormesh(dra, ddec, np.log10(slice_2d.value), cmap='inferno', shading='auto')