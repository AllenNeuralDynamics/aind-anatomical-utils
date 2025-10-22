import unittest

import numpy as np
import SimpleITK as sitk
from SimpleITK import DICOMOrientImageFilter

from aind_anatomical_utils import sitk_volume


def all_closer_than(a, b, thresh):
    return np.all(np.abs(a - b) <= thresh)


def fraction_close(a, val):
    arr = sitk.GetArrayViewFromImage(a)
    nel = np.prod(arr.shape)
    return np.sum(np.isclose(arr, val)) / nel


# Helper functions for regrid_axis_aligned tests
def create_test_image_with_orientation(
    coord_system: str,
    size: tuple[int, int, int] = (10, 20, 30),
    spacing: tuple[float, float, float] = (1.0, 2.0, 3.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pixel_type: int = sitk.sitkUInt8,
) -> sitk.Image:
    """Create a test SimpleITK image with specified orientation.

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
    pixel_type : int
        SimpleITK pixel type

    Returns
    -------
    sitk.Image
        Test image with specified parameters
    """
    img = sitk.Image(list(size), pixel_type)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)

    dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        coord_system
    )
    img.SetDirection(dir_tuple)

    return img


def create_gradient_image(
    coord_system: str,
    size: tuple[int, int, int] = (10, 20, 30),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> sitk.Image:
    """Create image with gradient pattern for testing reorientation.

    Creates an image where voxel value = i + 100*j + 10000*k, allowing
    us to identify which voxel is which after reorientation.

    Parameters
    ----------
    coord_system : str
        Orientation code
    size : tuple[int, int, int]
        Image size
    spacing : tuple[float, float, float]
        Voxel spacing
    origin : tuple[float, float, float]
        Image origin

    Returns
    -------
    sitk.Image
        Gradient test image
    """
    img = create_test_image_with_orientation(
        coord_system, size, spacing, origin, pixel_type=sitk.sitkInt32
    )

    # Create gradient pattern
    arr = np.zeros(size[::-1], dtype=np.int32)  # sitk uses ZYX ordering
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                # Store indices in a way we can recover them
                arr[k, j, i] = i + 100 * j + 10000 * k

    img_with_data = sitk.GetImageFromArray(arr)
    img_with_data.CopyInformation(img)

    return img_with_data


def get_voxel_value_at_physical_point(
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


def verify_physical_value_correspondence(
    img1: sitk.Image,
    idx1: tuple[int, int, int],
    img2: sitk.Image,
    idx2: tuple[int, int, int],
) -> bool:
    """Verify voxels at corresponding indices have same physical location and
    value.

    Parameters
    ----------
    img1 : sitk.Image
        First image
    idx1 : tuple[int, int, int]
        Index in first image
    img2 : sitk.Image
        Second image
    idx2 : tuple[int, int, int]
        Index in second image

    Returns
    -------
    bool
        True if physical locations match and values are equal
    """
    # Get physical points
    phys1 = img1.TransformIndexToPhysicalPoint(idx1)
    phys2 = img2.TransformIndexToPhysicalPoint(idx2)

    # Check physical locations match
    if not np.allclose(phys1, phys2, atol=1e-10):
        return False

    # Check values match
    val1 = img1.GetPixel(idx1)
    val2 = img2.GetPixel(idx2)

    return val1 == val2


class SITKTest(unittest.TestCase):
    test_index_translation_sets = [
        (np.array([[0, 0, 0], [2, 2, 2]]), np.array([[0, 0, 0], [2, 2, 2]])),
        (
            np.array([[0.5, 0.5, 0.5], [2, 2, 2]]),
            np.array([[0.5, 0.5, 0.5], [2, 2, 2]]),
        ),
    ]

    def test_transform_sitk_indices_to_physical_points(self) -> None:
        simg = sitk.Image(256, 128, 64, sitk.sitkUInt8)
        for ndxs, answer in self.test_index_translation_sets:
            received = sitk_volume.transform_sitk_indices_to_physical_points(
                simg, ndxs
            )
            self.assertTrue(np.allclose(answer, received))


if __name__ == "__main__":
    unittest.main()
