"""
Code to handle sitk volume loading and rotating

SimpleITK example code is under Apache License, see:
https://github.com/SimpleITK/TUTORIAL/blob/main/LICENSE

"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from aind_anatomical_utils.anatomical_volume import AnatomicalHeader


def transform_sitk_indices_to_physical_points(
    image: sitk.Image, index_arr: NDArray
) -> NDArray[np.floating]:
    """Transforms indices of image to physical points

    For a SimpleITK image `image` and a list of indices `index_arr`, transform
    each index to a physical point.

    Parameters
    ----------
    image : M-d SimpleITK image
    index_arr : numpy.ndarray (NxM)
        matrix of indices of `image`, where each row is an index

    Returns
    -------
    position_arr: numpy.ndarray (NxM)
        matrix of physical points for each index in `index_arr`
    """
    position_arr = np.zeros_like(index_arr, dtype="float32")
    npt = index_arr.shape[0]
    for pt_no in range(npt):
        ndx = tuple(map(lambda x: x.item(), index_arr[pt_no, :]))
        position_arr[pt_no, :] = image.TransformContinuousIndexToPhysicalPoint(
            ndx
        )
    return position_arr


def find_points_equal_to(
    image: sitk.Image, label_value: int | None = None
) -> NDArray[np.floating]:
    """
    Get the physical positions of all voxels in the implant volume that match
    the given label value.

    Parameters
    ----------
    image: SimpleITK.Image
        The implant volume to query.
    label_value : int or None
        The label value to search for in the volume. If None, the function
        returns non-zero positions.

    Returns
    -------
    ndarray
        A NumPy array of physical positions corresponding to the label value.
    """
    implant_vol_arr = sitk.GetArrayViewFromImage(image)
    if label_value is None:
        indices = np.nonzero(implant_vol_arr)
    else:
        indices = np.nonzero(implant_vol_arr == label_value)

    if len(indices[0]) == 0:
        return np.empty((0, implant_vol_arr.ndim))

    positions = [
        image.TransformIndexToPhysicalPoint(tuple([int(x) for x in idx[::-1]]))
        for idx in zip(*indices)
    ]
    return np.vstack(positions)


def regrid_axis_aligned_sitk(
    image: sitk.Image, dst_orientation: str
) -> sitk.Image:
    """
    Regrid a SimpleITK image to the specified axis-aligned orientation.

    Parameters
    ----------
    image : sitk.Image
        The input SimpleITK image to be regridded.
    dst_orientation : str
        The desired axis-aligned orientation code (e.g., 'RAS', 'LPI').

    Returns
    -------
    sitk.Image
        The regridded SimpleITK image in the specified orientation.
    """
    header = AnatomicalHeader.from_sitk(image)
    regridded_header = header.regrid_to(dst_orientation)
    ref_image = regridded_header.as_sitk(p_type=sitk.sitkUInt8)
    resampled_image = sitk.Resample(
        image,
        ref_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        image.GetPixelID(),
    )
    return resampled_image
