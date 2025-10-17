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


class TestRegridAxisAligned(unittest.TestCase):
    """Tests for regrid_axis_aligned function."""

    def test_regrid_RAS_to_LPS(self):
        """Test regridding from RAS to LPS with value verification."""
        img = create_gradient_image("RAS", size=(5, 6, 7))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Verify output orientation
        result_code = (
            DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                result.GetDirection()
            )
        )
        self.assertEqual(result_code, "LPS")

        # Verify a few voxel values at physical locations
        # Test corner voxel [0,0,0] in RAS
        src_phys = img.TransformIndexToPhysicalPoint((0, 0, 0))
        src_val = img.GetPixel((0, 0, 0))
        result_val = get_voxel_value_at_physical_point(result, src_phys)
        self.assertEqual(src_val, result_val)

    def test_regrid_LPS_to_RAS(self):
        """Test regridding from LPS to RAS."""
        img = create_gradient_image("LPS", size=(5, 6, 7))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "RAS")

        # Verify output orientation
        result_code = (
            DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                result.GetDirection()
            )
        )
        self.assertEqual(result_code, "RAS")

        # Verify values preserved at physical locations
        src_phys = img.TransformIndexToPhysicalPoint((2, 3, 4))
        src_val = img.GetPixel((2, 3, 4))
        result_val = get_voxel_value_at_physical_point(result, src_phys)
        self.assertEqual(src_val, result_val)

    def test_regrid_to_multiple_orientations(self):
        """Test regridding to various orientations."""
        orientations = ["RAS", "LPS", "RPI", "LAI"]
        src_img = create_gradient_image("RAS", size=(4, 5, 6))

        for dst_orient in orientations:
            with self.subTest(dst_orient=dst_orient):
                result = sitk_volume.regrid_axis_aligned_sitk(
                    src_img, dst_orient
                )

                # Verify output orientation
                result_code = (
                    DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                        result.GetDirection()
                    )
                )
                self.assertEqual(result_code, dst_orient)

                # Verify at least one voxel value preserved
                src_phys = src_img.TransformIndexToPhysicalPoint((1, 2, 3))
                src_val = src_img.GetPixel((1, 2, 3))
                result_val = get_voxel_value_at_physical_point(
                    result, src_phys
                )
                self.assertEqual(src_val, result_val)

    def test_regrid_identity_transformation(self):
        """Test regridding to same orientation preserves image exactly."""
        img = create_gradient_image("RAS", size=(5, 6, 7))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "RAS")

        # Should be identical
        self.assertEqual(img.GetSize(), result.GetSize())
        self.assertTrue(
            np.allclose(img.GetOrigin(), result.GetOrigin(), atol=1e-10)
        )
        self.assertTrue(
            np.allclose(img.GetSpacing(), result.GetSpacing(), atol=1e-10)
        )
        self.assertTrue(
            np.allclose(img.GetDirection(), result.GetDirection(), atol=1e-10)
        )

        # All voxel values should match
        arr_src = sitk.GetArrayViewFromImage(img)
        arr_result = sitk.GetArrayViewFromImage(result)
        self.assertTrue(np.array_equal(arr_src, arr_result))

    def test_regrid_round_trip_preserves_values(self):
        """Test RAS→LPS→RAS round trip preserves data."""
        original = create_gradient_image("RAS", size=(5, 6, 7))
        intermediate = sitk_volume.regrid_axis_aligned_sitk(original, "LPS")
        final = sitk_volume.regrid_axis_aligned_sitk(intermediate, "RAS")

        # Should return to original state
        self.assertEqual(original.GetSize(), final.GetSize())
        self.assertTrue(
            np.allclose(original.GetOrigin(), final.GetOrigin(), atol=1e-10)
        )
        self.assertTrue(
            np.allclose(original.GetSpacing(), final.GetSpacing(), atol=1e-10)
        )
        self.assertTrue(
            np.allclose(
                original.GetDirection(), final.GetDirection(), atol=1e-10
            )
        )

        # All voxel values should match
        arr_original = sitk.GetArrayViewFromImage(original)
        arr_final = sitk.GetArrayViewFromImage(final)
        self.assertTrue(np.array_equal(arr_original, arr_final))

    def test_corner_voxels_preserved_at_physical_locations(self):
        """Test that corner voxel values are at correct physical locations."""
        img = create_gradient_image("RAS", size=(5, 6, 7))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Test all 8 corners
        size = img.GetSize()
        corners = [
            (0, 0, 0),
            (0, 0, size[2] - 1),
            (0, size[1] - 1, 0),
            (0, size[1] - 1, size[2] - 1),
            (size[0] - 1, 0, 0),
            (size[0] - 1, 0, size[2] - 1),
            (size[0] - 1, size[1] - 1, 0),
            (size[0] - 1, size[1] - 1, size[2] - 1),
        ]

        for corner in corners:
            src_phys = img.TransformIndexToPhysicalPoint(corner)
            src_val = img.GetPixel(corner)
            result_val = get_voxel_value_at_physical_point(result, src_phys)
            self.assertEqual(
                src_val,
                result_val,
                f"Corner {corner} mismatch: {src_val} != {result_val}",
            )

    def test_center_voxel_preserved(self):
        """Test that center voxel value remains at center after regridding."""
        img = create_gradient_image("RAS", size=(10, 20, 30))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPI")

        # Test center voxel
        center_idx = (5, 10, 15)
        src_phys = img.TransformIndexToPhysicalPoint(center_idx)
        src_val = img.GetPixel(center_idx)
        result_val = get_voxel_value_at_physical_point(result, src_phys)
        self.assertEqual(src_val, result_val)

    def test_gradient_pattern_correctly_reoriented(self):
        """Create a gradient image and verify pattern after reorientation."""
        img = create_gradient_image("RAS", size=(3, 4, 5))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Sample several voxels and verify their values match at physical
        # locations
        test_indices = [(0, 0, 0), (1, 2, 3), (2, 3, 4)]

        for idx in test_indices:
            src_phys = img.TransformIndexToPhysicalPoint(idx)
            src_val = img.GetPixel(idx)
            result_val = get_voxel_value_at_physical_point(result, src_phys)
            self.assertEqual(src_val, result_val, f"Mismatch at index {idx}")

    def test_output_has_correct_orientation(self):
        """Verify output image has the requested orientation code."""
        test_cases = [
            ("RAS", "LPS"),
            ("LPS", "RAS"),
            ("RAS", "RPI"),
            ("LPS", "LAI"),
        ]

        for src_orient, dst_orient in test_cases:
            with self.subTest(src=src_orient, dst=dst_orient):
                img = create_test_image_with_orientation(src_orient)
                result = sitk_volume.regrid_axis_aligned_sitk(img, dst_orient)

                result_code = (
                    DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                        result.GetDirection()
                    )
                )
                self.assertEqual(result_code, dst_orient)

    def test_output_size_spacing_correct(self):
        """Verify size and spacing are correctly permuted."""
        img = create_test_image_with_orientation(
            "RAS", size=(10, 20, 30), spacing=(1.0, 2.0, 3.0)
        )
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # For RAS to LPS, axes are flipped but not permuted
        # Size should be permuted according to axis mapping
        result_size = result.GetSize()
        result_spacing = result.GetSpacing()

        # Verify physical extents are preserved
        src_extents = np.array(img.GetSize()) * np.array(img.GetSpacing())
        result_extents = np.array(result_size) * np.array(result_spacing)

        # Sort both to compare (order may differ)
        self.assertTrue(
            np.allclose(
                sorted(src_extents), sorted(result_extents), atol=1e-10
            )
        )

    def test_output_preserves_pixel_type(self):
        """Verify output pixel type matches input."""
        pixel_types = [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkFloat32]

        for ptype in pixel_types:
            with self.subTest(pixel_type=ptype):
                img = create_test_image_with_orientation(
                    "RAS", pixel_type=ptype
                )
                result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")
                self.assertEqual(img.GetPixelID(), result.GetPixelID())

    def test_regrid_with_uint8(self):
        """Test with uint8 pixel type."""
        img = create_test_image_with_orientation(
            "RAS", pixel_type=sitk.sitkUInt8
        )
        # Set some values
        img.SetPixel((0, 0, 0), 42)
        img.SetPixel((1, 1, 1), 100)

        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Verify values preserved
        phys0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
        val0 = get_voxel_value_at_physical_point(result, phys0)
        self.assertEqual(val0, 42)

    def test_regrid_with_uint16(self):
        """Test with uint16 pixel type."""
        img = create_test_image_with_orientation(
            "RAS", pixel_type=sitk.sitkUInt16
        )
        img.SetPixel((0, 0, 0), 1000)
        img.SetPixel((1, 1, 1), 2000)

        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        phys0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
        val0 = get_voxel_value_at_physical_point(result, phys0)
        self.assertEqual(val0, 1000)

    def test_regrid_with_float32(self):
        """Test with float32 pixel type."""
        img = create_test_image_with_orientation(
            "RAS", pixel_type=sitk.sitkFloat32
        )
        img.SetPixel((0, 0, 0), 3.14)
        img.SetPixel((1, 1, 1), 2.71)

        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        phys0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
        val0 = get_voxel_value_at_physical_point(result, phys0)
        self.assertAlmostEqual(val0, 3.14, places=5)

    def test_regrid_with_anisotropic_spacing(self):
        """Test with non-uniform voxel spacing."""
        img = create_gradient_image(
            "RAS", size=(5, 6, 7), spacing=(0.5, 1.0, 2.0)
        )
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Verify values preserved at physical locations
        test_indices = [(0, 0, 0), (2, 3, 4), (4, 5, 6)]
        for idx in test_indices:
            src_phys = img.TransformIndexToPhysicalPoint(idx)
            src_val = img.GetPixel(idx)
            result_val = get_voxel_value_at_physical_point(result, src_phys)
            self.assertEqual(src_val, result_val)

    def test_regrid_with_non_zero_origin(self):
        """Test with various origin locations."""
        origins = [
            (0.0, 0.0, 0.0),
            (100.0, -50.0, 25.0),
            (-10.0, -20.0, -30.0),
        ]

        for origin in origins:
            with self.subTest(origin=origin):
                img = create_gradient_image(
                    "RAS", size=(4, 5, 6), origin=origin
                )
                result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

                # Verify values preserved
                idx = (1, 2, 3)
                src_phys = img.TransformIndexToPhysicalPoint(idx)
                src_val = img.GetPixel(idx)
                result_val = get_voxel_value_at_physical_point(
                    result, src_phys
                )
                self.assertEqual(src_val, result_val)

    def test_regrid_small_volume(self):
        """Test with 2x2x2 volume."""
        img = create_gradient_image("RAS", size=(2, 2, 2))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Verify all 8 voxels
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    idx = (i, j, k)
                    src_phys = img.TransformIndexToPhysicalPoint(idx)
                    src_val = img.GetPixel(idx)
                    result_val = get_voxel_value_at_physical_point(
                        result, src_phys
                    )
                    self.assertEqual(src_val, result_val)

    def test_regrid_thin_slice(self):
        """Test with thin slice (256x256x1)."""
        img = create_gradient_image("RAS", size=(16, 16, 1))
        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Verify corners
        corners = [(0, 0, 0), (15, 0, 0), (0, 15, 0), (15, 15, 0)]
        for corner in corners:
            src_phys = img.TransformIndexToPhysicalPoint(corner)
            src_val = img.GetPixel(corner)
            result_val = get_voxel_value_at_physical_point(result, src_phys)
            self.assertEqual(src_val, result_val)

    def test_regrid_non_axis_aligned_raises_error(self):
        """Test that non-axis-aligned images raise ValueError."""
        # Create oblique image
        img = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        angle = np.pi / 6
        direction = [
            np.cos(angle),
            -np.sin(angle),
            0.0,
            np.sin(angle),
            np.cos(angle),
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        img.SetDirection(direction)

        with self.assertRaises(ValueError) as ctx:
            sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        self.assertIn("axis-aligned", str(ctx.exception).lower())

    def test_regrid_labeled_volume_preserves_labels(self):
        """Create volume with distinct labeled regions, verify after regrid."""
        img = create_test_image_with_orientation(
            "RAS", size=(10, 10, 10), pixel_type=sitk.sitkUInt8
        )

        # Create labeled regions
        # Region 1: corner [0-4, 0-4, 0-4] = label 1
        # Region 2: corner [5-9, 5-9, 5-9] = label 2
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    img.SetPixel((i, j, k), 1)
                    img.SetPixel((i + 5, j + 5, k + 5), 2)

        result = sitk_volume.regrid_axis_aligned_sitk(img, "LPS")

        # Verify labels preserved at physical locations
        # Test center of each region
        region1_center = (2, 2, 2)
        region2_center = (7, 7, 7)

        phys1 = img.TransformIndexToPhysicalPoint(region1_center)
        phys2 = img.TransformIndexToPhysicalPoint(region2_center)

        val1 = get_voxel_value_at_physical_point(result, phys1)
        val2 = get_voxel_value_at_physical_point(result, phys2)

        self.assertEqual(val1, 1)
        self.assertEqual(val2, 2)


if __name__ == "__main__":
    unittest.main()
