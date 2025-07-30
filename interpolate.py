import numpy as np
import astropy.units as u
from scipy.interpolate import griddata

def interpolate_pixels(slice, target_value=1e-32, method='linear', fallback='nearest'):
    """
    Interpolates pixels in a 2D array (units uJy) that are equal to a specific target value.
    
    Parameters:
        data2d : np.ndarray
            The 2D input array. Needs .value or it won't work
        target_value : float
            The pixel value to be treated as 'missing' and replaced via interpolation.
        method : str
            Primary interpolation method ('linear', 'nearest', 'cubic').
        fallback : str or None
            Fallback method to try if primary interpolation fails at a pixel.

    Returns:
        interpolated_map : np.ndarray
            The same shape as input, with target_value pixels replaced via interpolation.
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
