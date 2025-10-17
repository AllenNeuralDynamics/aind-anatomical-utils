from __future__ import annotations

import ants  # type: ignore[import-untyped]
import numpy as np
from ants.core import ANTsImage  # type: ignore[import-untyped]

from aind_anatomical_utils.anatomical_volume import AnatomicalHeader
from aind_anatomical_utils.coordinate_systems import (
    find_coordinate_perm_and_flips,
)


def regrid_axis_aligned_ants(
    image: ANTsImage, dst_orientation: str
) -> ANTsImage:
    """
    Regrid an ANTs image to the specified axis-aligned orientation.

    Parameters
    ----------
    image : ANTsImage
        The input ANTs image to be regridded.
    dst_orientation : str
        The desired axis-aligned orientation code (e.g., 'RAS', 'LPI').

    Returns
    -------
    ANTsImage
        The regridded ANTs image in the specified orientation.
    """
    header = AnatomicalHeader.from_ants(image)
    # Ants will reverse the numpy array axes, but it is undone when creating
    # the ANTs image, so we can ignore it
    image_arr = image.numpy()
    perms, signs = find_coordinate_perm_and_flips(
        src=header.orientation_code(), dst=dst_orientation
    )
    flips = np.array(signs == -1, dtype=bool)
    axes_to_flip = np.nonzero(flips)[0]
    image_arr_regridded = np.permute_dims(image_arr, perms)
    if len(axes_to_flip) > 0:
        image_arr_regridded = np.flip(
            image_arr_regridded, axis=tuple(axes_to_flip)
        )
    regridded_header = header.regrid_to(dst_orientation)
    regridded_img = ants.from_numpy(image_arr_regridded)
    regridded_header.update_ants(regridded_img)
    return regridded_img
