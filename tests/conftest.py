"""Shared test utilities for regrid functions across libraries."""

import ants  # type: ignore[import-untyped]
import numpy as np
import SimpleITK as sitk
from ants.core import ANTsImage
from SimpleITK import DICOMOrientImageFilter

# ============================================================================
# Shared Gradient Pattern Creation
# ============================================================================


def create_gradient_sitk_image(
    coord_system: str,
    size: tuple[int, int, int] = (10, 20, 30),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> sitk.Image:
    """Create SimpleITK image with gradient pattern for testing reorientation.

    Creates an image where voxel value = i + 100*j + 10000*k, allowing
    us to identify which voxel is which after reorientation.

    Parameters
    ----------
    coord_system : str
        Orientation code (e.g., 'RAS', 'LPS')
    size : tuple[int, int, int]
        Image size (i, j, k)
    spacing : tuple[float, float, float]
        Voxel spacing
    origin : tuple[float, float, float]
        Image origin in LPS coordinates

    Returns
    -------
    sitk.Image
        Gradient test image
    """
    # Create image with correct orientation
    img = sitk.Image(list(size), sitk.sitkInt32)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    # Set direction based on orientation code
    dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        coord_system
    )
    img.SetDirection(dir_tuple)

    # Create gradient pattern: value = i + 100*j + 10000*k
    arr = np.zeros(size[::-1], dtype=np.int32)  # SimpleITK uses ZYX ordering
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                arr[k, j, i] = i + 100 * j + 10000 * k

    img_with_data = sitk.GetImageFromArray(arr)
    img_with_data.CopyInformation(img)

    return img_with_data


def create_gradient_ants_image(
    coord_system: str,
    size: tuple[int, int, int] = (10, 20, 30),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ANTsImage:
    """Create ANTs image with gradient pattern for testing reorientation.

    Creates an image where voxel value = i + 100*j + 10000*k, allowing
    us to identify which voxel is which after reorientation.

    Parameters
    ----------
    coord_system : str
        Orientation code (e.g., 'RAS', 'LPS')
    size : tuple[int, int, int]
        Image size (i, j, k)
    spacing : tuple[float, float, float]
        Voxel spacing
    origin : tuple[float, float, float]
        Image origin

    Returns
    -------
    ANTsImage
        Gradient test image
    """
    # Create gradient pattern: value = i + 100*j + 10000*k
    arr = np.zeros(size, dtype=np.int32)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                arr[i, j, k] = i + 100 * j + 10000 * k

    # Get direction matrix for orientation
    dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        coord_system
    )
    direction = np.array(dir_tuple).reshape(3, 3)

    # Create ANTs image with correct header
    img = ants.from_numpy(
        arr, origin=origin, spacing=spacing, direction=direction
    )

    return img


# ============================================================================
# Shared Verification Helpers
# ============================================================================


def get_sitk_voxel_value_at_physical_point(
    img: sitk.Image, physical_point: tuple[float, float, float]
) -> float:
    """Get the voxel value at a physical point using nearest neighbor.

    Parameters
    ----------
    img : sitk.Image
        Input image
    physical_point : tuple[float, float, float]
        Physical coordinates (LPS)

    Returns
    -------
    float
        Voxel value at that location
    """
    # Transform physical point to continuous index
    continuous_idx = img.TransformPhysicalPointToContinuousIndex(
        physical_point
    )
    # Round to nearest integer index
    idx = tuple(int(round(x)) for x in continuous_idx)

    # Check bounds
    size = img.GetSize()
    if not all(0 <= idx[i] < size[i] for i in range(3)):
        raise ValueError(f"Index {idx} out of bounds for size {size}")

    return float(img.GetPixel(idx))


def get_ants_voxel_value_at_physical_point(
    img: ANTsImage, physical_point: tuple[float, float, float]
) -> float:
    """Get the voxel value at a physical point using nearest neighbor.

    Parameters
    ----------
    img : ANTsImage
        Input image
    physical_point : tuple[float, float, float]
        Physical coordinates

    Returns
    -------
    float
        Voxel value at that location
    """
    # Transform physical point to index
    # physical = origin + direction @ (spacing * index)
    # Solve for index: index = (direction^-1 @ (physical - origin)) / spacing
    point_arr = np.array(physical_point)
    origin_arr = np.array(img.origin)
    direction_inv = np.linalg.inv(img.direction)
    spacing_arr = np.array(img.spacing)

    index_continuous = direction_inv @ (point_arr - origin_arr) / spacing_arr

    # Round to nearest integer index
    idx = tuple(int(round(x)) for x in index_continuous)

    # Check bounds
    size = img.shape
    if not all(0 <= idx[i] < size[i] for i in range(3)):
        raise ValueError(f"Index {idx} out of bounds for size {size}")

    # Get pixel value
    arr = img.numpy()
    return float(arr[idx[0], idx[1], idx[2]])


def get_ants_orientation_code(img: ANTsImage) -> str:
    """Get the orientation code for an ANTs image.

    Parameters
    ----------
    img : ANTsImage
        Input image

    Returns
    -------
    str
        Orientation code (e.g., 'RAS', 'LPS')
    """
    from aind_anatomical_utils.anatomical_volume import AnatomicalHeader

    header = AnatomicalHeader.from_ants(img)
    return header.orientation_code()
