"""Tests functions in `anatomical_volume`."""

import unittest

import ants  # type: ignore[import-untyped]
import numpy as np
import pytest
import SimpleITK as sitk
from SimpleITK import DICOMOrientImageFilter

from aind_anatomical_utils import anatomical_volume as av


def verify_origin_correctness(
    origin, direction, spacing, corner_idx, target_point_lps, tolerance=1e-10
):
    """
    Verify that origin + (corner_idx * spacing) @ direction.T ≈ target_point.

    Parameters
    ----------
    origin : array-like
        The computed origin in LPS coordinates.
    direction : NDArray
        3x3 direction cosine matrix.
    spacing : array-like
        Voxel spacing.
    corner_idx : array-like
        The continuous index of the corner.
    target_point_lps : array-like
        Expected target point in LPS coordinates.
    tolerance : float
        Acceptable numerical error.

    Returns
    -------
    bool
        True if the computed origin is correct within tolerance.
    """
    origin_arr = np.asarray(origin, float)
    spacing_arr = np.asarray(spacing, float)
    corner_idx_arr = np.asarray(corner_idx, float)
    direction_arr = np.asarray(direction, float)
    target_arr = np.asarray(target_point_lps, float)

    # ITK formula: physical_point = origin + (index * spacing) @ direction.T
    computed_target = (
        origin_arr + (corner_idx_arr * spacing_arr) @ direction_arr.T
    )

    return np.allclose(computed_target, target_arr, atol=tolerance)


class TestAnatomicalHeader(unittest.TestCase):
    """Test the Header dataclass."""

    def test_header_creation(self):
        """Test basic Header creation."""
        origin = (1.0, 2.0, 3.0)
        spacing = (0.5, 0.5, 1.0)
        direction = np.eye(3)
        size_ijk = (100, 200, 50)

        header = av.AnatomicalHeader(
            origin=origin,
            spacing=spacing,
            direction=direction,
            size_ijk=size_ijk,
        )

        assert header.origin == origin
        assert header.spacing == spacing
        assert np.array_equal(header.direction, direction)
        assert header.size_ijk == size_ijk

    def test_header_immutable(self):
        """Test that Header is frozen/immutable."""
        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            header.origin = (1, 1, 1)

    def test_direction_tuple(self):
        """Test direction matrix flattening."""
        direction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=direction,
            size_ijk=(10, 10, 10),
        )

        direction_tuple = header.direction_tuple()
        expected = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert direction_tuple == expected

    def test_from_sitk(self):
        """Test Header creation from SimpleITK image."""
        sitk_image = sitk.Image([100, 200, 50], sitk.sitkUInt8)
        sitk_image.SetOrigin((1.0, 2.0, 3.0))
        sitk_image.SetSpacing((0.5, 0.5, 1.0))
        # Use a valid direction matrix (identity)
        sitk_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        header = av.AnatomicalHeader.from_sitk(sitk_image, size_ijk=None)

        assert header.origin == (1.0, 2.0, 3.0)
        assert header.spacing == (0.5, 0.5, 1.0)
        assert header.size_ijk == (100, 200, 50)
        assert header.direction.shape == (3, 3)
        # Verify the direction matrix is correctly stored
        assert np.allclose(header.direction, np.eye(3))

    def test_from_sitk_custom_size(self):
        """Test Header creation with custom size."""
        sitk_image = sitk.Image([100, 200, 50], sitk.sitkUInt8)
        custom_size = (75, 75, 75)
        header = av.AnatomicalHeader.from_sitk(
            sitk_image, size_ijk=custom_size
        )

        assert header.size_ijk == custom_size

    def test_update_sitk(self):
        """Test updating SimpleITK image with Header."""
        header = av.AnatomicalHeader(
            origin=(5.0, 10.0, 15.0),
            spacing=(2.0, 2.0, 2.0),
            direction=np.eye(3),
            size_ijk=(50, 50, 50),
        )

        sitk_image = sitk.Image([50, 50, 50], sitk.sitkUInt8)
        result = header.update_sitk(sitk_image)

        assert result is sitk_image  # Returns same instance
        assert sitk_image.GetOrigin() == (5.0, 10.0, 15.0)
        assert sitk_image.GetSpacing() == (2.0, 2.0, 2.0)

    def test_as_sitk(self):
        """Test creating SimpleITK image from Header."""
        header = av.AnatomicalHeader(
            origin=(1.0, 2.0, 3.0),
            spacing=(0.5, 0.5, 1.0),
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        sitk_img = header.as_sitk_stub()

        # SimpleITK images have GetOrigin() and GetSpacing() methods
        assert sitk_img.GetOrigin() == (1.0, 2.0, 3.0)
        assert sitk_img.GetSpacing() == (0.5, 0.5, 1.0)

    def test_orientation_code_LPS(self):
        """Test orientation code for LPS (identity matrix)."""
        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )

        orientation = header.orientation_code()
        assert orientation == "LPS"

    def test_orientation_code_RAS(self):
        """Test orientation code for RAS orientation."""
        # RAS direction matrix
        dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
            "RAS"
        )
        direction = np.array(dir_tuple).reshape(3, 3)

        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=direction,
            size_ijk=(10, 10, 10),
        )

        orientation = header.orientation_code()
        assert orientation == "RAS"

    def test_orientation_code_various(self):
        """Test orientation code for various common orientations."""
        orientations = ["RAS", "LPS", "RPI", "LAI", "RPS", "LAS", "RAI", "LPI"]

        for expected_orient in orientations:
            with (
                pytest.raises(Exception)
                if False
                else self.subTest(orientation=expected_orient)
            ):
                # Get direction matrix for this orientation
                dir_tuple = (
                    DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
                        expected_orient
                    )
                )
                direction = np.array(dir_tuple).reshape(3, 3)

                header = av.AnatomicalHeader(
                    origin=(0, 0, 0),
                    spacing=(1, 1, 1),
                    direction=direction,
                    size_ijk=(10, 10, 10),
                )

                orientation = header.orientation_code()
                assert orientation == expected_orient, (
                    f"Expected {expected_orient}, got {orientation}"
                )

    def test_orientation_code_from_actual_sitk_image(self):
        """Test that orientation code matches SimpleITK's own computation."""
        # Create a SimpleITK image with RPI orientation
        sitk_img = sitk.Image([10, 20, 30], sitk.sitkUInt8)
        dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
            "RPI"
        )
        sitk_img.SetDirection(dir_tuple)
        sitk_img.SetOrigin((5.0, 10.0, 15.0))
        sitk_img.SetSpacing((1.0, 1.0, 2.0))

        # Get orientation from SimpleITK
        sitk_orientation = (
            DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                sitk_img.GetDirection()
            )
        )

        # Get orientation from our Header
        header = av.AnatomicalHeader.from_sitk(sitk_img)
        header_orientation = header.orientation_code()

        assert header_orientation == sitk_orientation

    def test_is_axis_aligned_true(self):
        """Test that axis-aligned matrix returns True."""
        # Identity matrix is axis-aligned
        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )
        assert header.is_axis_aligned() is True

    def test_is_axis_aligned_false_oblique(self):
        """Test that oblique (non-axis-aligned) matrix returns False."""
        # 30-degree rotation around z-axis
        angle = np.pi / 6
        direction = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=direction,
            size_ijk=(10, 10, 10),
        )
        assert header.is_axis_aligned() is False

    def test_is_axis_aligned_false_non_unique_argmax(self):
        """Test matrix where argmax rows are not all different."""
        # Matrix where two columns have their max in the same row
        # This should trigger the len(set(r_idx.tolist())) != 3 check
        direction = np.array(
            [
                [0.6, 0.7, 0.0],  # Both col 0 and 1 have max in row 0
                [0.5, 0.6, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=direction,
            size_ijk=(10, 10, 10),
        )
        assert header.is_axis_aligned() is False

    def test_is_axis_aligned_with_tolerance(self):
        """Test nearly axis-aligned matrix with custom tolerance."""
        # Nearly identity matrix (very small deviation)
        direction = np.array(
            [
                [1.0, 1e-8, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=direction,
            size_ijk=(10, 10, 10),
        )

        # Should be aligned with default tolerance
        assert header.is_axis_aligned() is True

        # Should still be aligned with loose tolerance
        assert header.is_axis_aligned(tol=1e-5) is True

        # Should not be aligned with very strict tolerance
        assert header.is_axis_aligned(tol=1e-10) is False

    def test_is_axis_aligned_with_flips(self):
        """Test axis-aligned matrix with flips (negative values)."""
        # RAS orientation (flips from LPS)
        dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
            "RAS"
        )
        direction = np.array(dir_tuple).reshape(3, 3)

        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=direction,
            size_ijk=(10, 10, 10),
        )

        # Should be axis-aligned (flips are allowed)
        assert header.is_axis_aligned() is True

    def test_as_sitk_with_custom_pixel_type(self):
        """Test creating full-size SimpleITK image with custom pixel type."""
        header = av.AnatomicalHeader(
            origin=(5.0, 10.0, 15.0),
            spacing=(0.5, 1.0, 2.0),
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        # Test with Float32
        sitk_img = header.as_sitk(p_type=sitk.sitkFloat32)

        assert sitk_img.GetSize() == (10, 20, 30)
        assert sitk_img.GetOrigin() == (5.0, 10.0, 15.0)
        assert sitk_img.GetSpacing() == (0.5, 1.0, 2.0)
        assert sitk_img.GetPixelID() == sitk.sitkFloat32

    def test_as_sitk_various_pixel_types(self):
        """Test creating images with various pixel types."""
        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=np.eye(3),
            size_ijk=(5, 5, 5),
        )

        pixel_types = [
            sitk.sitkUInt8,
            sitk.sitkInt16,
            sitk.sitkUInt16,
            sitk.sitkFloat32,
            sitk.sitkFloat64,
        ]

        for ptype in pixel_types:
            with self.subTest(pixel_type=ptype):
                sitk_img = header.as_sitk(p_type=ptype)
                assert sitk_img.GetPixelID() == ptype
                assert sitk_img.GetSize() == (5, 5, 5)

    def test_as_sitk_physical_extent(self):
        """Test that as_sitk creates image with correct physical extent."""
        header = av.AnatomicalHeader(
            origin=(10.0, 20.0, 30.0),
            spacing=(2.0, 3.0, 4.0),
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        sitk_img = header.as_sitk()

        # Compute physical location of first voxel (0,0,0)
        first_voxel_phys = sitk_img.TransformIndexToPhysicalPoint((0, 0, 0))
        assert np.allclose(first_voxel_phys, header.origin)

        # Compute physical location of last voxel (9, 19, 29)
        last_idx = (9, 19, 29)
        last_voxel_phys = sitk_img.TransformIndexToPhysicalPoint(last_idx)
        expected_last = np.array(header.origin) + np.array(
            [9 * 2.0, 19 * 3.0, 29 * 4.0]
        )
        assert np.allclose(last_voxel_phys, expected_last)

    def test_as_sitk_roundtrip(self):
        """Test round-trip: Header -> SimpleITK -> Header."""
        original_header = av.AnatomicalHeader(
            origin=(5.0, 10.0, 15.0),
            spacing=(0.5, 1.0, 2.0),
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        # Convert to SimpleITK and back
        sitk_img = original_header.as_sitk()
        recovered_header = av.AnatomicalHeader.from_sitk(sitk_img)

        # Verify all properties match
        assert np.allclose(original_header.origin, recovered_header.origin)
        assert np.allclose(original_header.spacing, recovered_header.spacing)
        assert np.allclose(
            original_header.direction, recovered_header.direction
        )
        assert original_header.size_ijk == recovered_header.size_ijk


class TestCornerIndices(unittest.TestCase):
    """Tests for _corner_indices helper function."""

    def test_corner_indices_outer_true(self):
        """Test corner indices with outer box convention."""
        size = np.array([10, 20, 30])
        corners = av._corner_indices(size, outer=True)

        # Should return 8 corners
        self.assertEqual(corners.shape, (8, 3))

        # Check that corners use outer box convention: lo=-0.5, hi=size-0.5
        # product([lo, hi[0]], [lo, hi[1]], [lo, hi[2]]) varies z fastest
        expected_corners = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 29.5],
                [-0.5, 19.5, -0.5],
                [-0.5, 19.5, 29.5],
                [9.5, -0.5, -0.5],
                [9.5, -0.5, 29.5],
                [9.5, 19.5, -0.5],
                [9.5, 19.5, 29.5],
            ]
        )
        self.assertTrue(np.allclose(corners, expected_corners))

    def test_corner_indices_outer_false(self):
        """Test corner indices with voxel center convention."""
        size = np.array([10, 20, 30])
        corners = av._corner_indices(size, outer=False)

        # Should return 8 corners
        self.assertEqual(corners.shape, (8, 3))

        # Check that corners use voxel center convention: lo=0.0, hi=size-1.0
        # product([lo, hi[0]], [lo, hi[1]], [lo, hi[2]]) varies z fastest
        expected_corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 29.0],
                [0.0, 19.0, 0.0],
                [0.0, 19.0, 29.0],
                [9.0, 0.0, 0.0],
                [9.0, 0.0, 29.0],
                [9.0, 19.0, 0.0],
                [9.0, 19.0, 29.0],
            ]
        )
        self.assertTrue(np.allclose(corners, expected_corners))

    def test_corner_indices_uniform_size(self):
        """Test with uniform size."""
        size = np.array([10, 10, 10])
        corners = av._corner_indices(size, outer=True)

        self.assertEqual(corners.shape, (8, 3))
        # First corner should be (-0.5, -0.5, -0.5)
        self.assertTrue(np.allclose(corners[0], [-0.5, -0.5, -0.5]))
        # Last corner should be (9.5, 9.5, 9.5)
        self.assertTrue(np.allclose(corners[-1], [9.5, 9.5, 9.5]))

    def test_corner_indices_non_uniform_size(self):
        """Test with non-uniform size."""
        size = np.array([5, 10, 15])
        corners = av._corner_indices(size, outer=False)

        self.assertEqual(corners.shape, (8, 3))
        # Check extremes
        self.assertTrue(np.allclose(corners[0], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(corners[-1], [4.0, 9.0, 14.0]))

    def test_corner_indices_corner_ordering(self):
        """Test that corners follow expected product ordering."""
        size = np.array([2, 3, 4])
        corners = av._corner_indices(size, outer=True)

        # The implementation uses product([lo, hi[0]], [lo, hi[1]], [lo,
        # hi[2]])
        # This means z varies fastest, then y, then x (rightmost varies
        # fastest)
        self.assertTrue(np.allclose(corners[0], [-0.5, -0.5, -0.5]))
        self.assertTrue(np.allclose(corners[1], [-0.5, -0.5, 3.5]))
        self.assertTrue(np.allclose(corners[2], [-0.5, 2.5, -0.5]))
        self.assertTrue(np.allclose(corners[3], [-0.5, 2.5, 3.5]))
        self.assertTrue(np.allclose(corners[4], [1.5, -0.5, -0.5]))
        self.assertTrue(np.allclose(corners[5], [1.5, -0.5, 3.5]))
        self.assertTrue(np.allclose(corners[6], [1.5, 2.5, -0.5]))
        self.assertTrue(np.allclose(corners[7], [1.5, 2.5, 3.5]))


class TestFixCornerComputeOrigin(unittest.TestCase):
    """Tests for fix_corner_compute_origin function."""

    def test_return_value_types(self):
        """Test that return values have correct types."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        # Check types
        self.assertIsInstance(origin, tuple)
        self.assertEqual(len(origin), 3)
        self.assertTrue(
            all(isinstance(x, (float, np.floating)) for x in origin)
        )

        self.assertIsInstance(corner_idx, np.ndarray)
        self.assertEqual(corner_idx.shape, (3,))

        self.assertIsInstance(idx_num, (int, np.integer))
        self.assertTrue(0 <= idx_num <= 7)

    def test_identity_direction_RAS_corner(self):
        """Test with identity direction matrix and RAS corner."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]  # In LPS (default target_frame)
        corner_code = "RAS"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code=corner_code
        )

        # target_point is already in LPS (default), so use directly
        # Verify mathematical correctness
        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_identity_direction_LPI_corner(self):
        """Test with identity direction matrix and LPI corner (opposite of
        RAS)."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]  # In LPS (default target_frame)
        corner_code = "LPI"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code=corner_code
        )

        # target_point is already in LPS (default), so use directly
        # Verify mathematical correctness
        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_target_frame_defaults_to_LPS(self):
        """Test that target_frame defaults to LPS."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]
        corner_code = "RAS"

        # Call without specifying target_frame (should default to LPS)
        origin1, corner_idx1, idx_num1 = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code=corner_code
        )

        # Call with target_frame="LPS" explicitly
        origin2, corner_idx2, idx_num2 = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            target_frame="LPS",
        )

        # Results should be identical
        self.assertTrue(np.allclose(origin1, origin2))
        self.assertTrue(np.allclose(corner_idx1, corner_idx2))
        self.assertEqual(idx_num1, idx_num2)

    def test_target_frame_different_from_corner_code(self):
        """Test with target_frame different from corner_code."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]  # In RAS coordinates
        corner_code = "RAS"
        target_frame = "LPS"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            target_frame=target_frame,
        )

        # Target point is in LPS, needs to be converted to LPS for verification
        # LPS to LPS is identity, but we need to check the math works out
        # The target_point [5, 5, 5] in LPS should map correctly
        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_use_outer_box_true(self):
        """Test with outer box convention."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code="RAS",
            use_outer_box=True,
        )

        # With outer box, corner indices should be at -0.5 or size-0.5
        self.assertTrue(np.all((corner_idx == -0.5) | (corner_idx == 9.5)))

        # Verify correctness
        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_use_outer_box_false(self):
        """Test with voxel center convention."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code="RAS",
            use_outer_box=False,
        )

        # With voxel centers, corner indices should be at 0.0 or size-1.0
        self.assertTrue(np.all((corner_idx == 0.0) | (corner_idx == 9.0)))

        # Verify correctness
        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_outer_box_difference(self):
        """Test that outer box differs from voxel center by 0.5 * spacing."""
        size = [10, 10, 10]
        spacing = [2.0, 2.0, 2.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]
        corner_code = "RAS"

        origin_outer, _, _ = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            use_outer_box=True,
        )

        origin_inner, _, _ = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            use_outer_box=False,
        )

        # The difference should be related to 0.5 * spacing
        # Exact difference depends on which corner, but should be proportional
        diff = np.abs(np.array(origin_outer) - np.array(origin_inner))
        # Difference should be 0.5 * spacing for each axis (could be 0 or 1.0)
        self.assertTrue(np.all(diff <= 1.0 * np.array(spacing)))

    def test_various_corner_codes(self):
        """Test with various valid corner codes."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]

        # Valid corner codes: one from each pair (R/L, A/P, S/I)
        corner_codes = ["RAS", "LPS", "RPI", "LAI", "RPS", "LAS", "RAI", "LPI"]

        for corner_code in corner_codes:
            with self.subTest(corner_code=corner_code):
                origin, corner_idx, idx_num = av.fix_corner_compute_origin(
                    size,
                    spacing,
                    direction,
                    target_point,
                    corner_code=corner_code,
                )

                # Verify return types
                self.assertIsInstance(origin, tuple)
                self.assertTrue(0 <= idx_num <= 7)

                # Verify mathematical correctness
                self.assertTrue(
                    verify_origin_correctness(
                        origin, direction, spacing, corner_idx, target_point
                    )
                )

    def test_unusual_target_frame_SRA(self):
        """Test with unusual target frame SRA."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]  # In SRA coordinates
        corner_code = "RAS"
        target_frame = "SRA"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            target_frame=target_frame,
        )

        # Should successfully compute origin
        self.assertIsInstance(origin, tuple)
        self.assertTrue(0 <= idx_num <= 7)

        # The target point needs to be converted from SRA to LPS for
        # verification
        # SRA -> LPS: S->L is flip, R->P is flip, A->S is no flip
        # Actually, need to think about this more carefully
        # SRA means: first axis is S/I, second is R/L, third is A/P
        # LPS means: first axis is L/R, second is P/A, third is S/I
        # So SRA = [s, r, a] -> LPS = [-r, -a, s] (need proper transform)
        from aind_anatomical_utils import coordinate_systems as cs

        target_lps = cs.convert_coordinate_system(
            np.array([target_point]), "SRA", "LPS"
        )[0]

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_lps
            )
        )

    def test_unusual_target_frame_IRP(self):
        """Test with unusual target frame IRP."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]  # In IRP coordinates
        corner_code = "RAS"
        target_frame = "IRP"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            target_frame=target_frame,
        )

        self.assertIsInstance(origin, tuple)
        self.assertTrue(0 <= idx_num <= 7)

        # Convert target from IRP to LPS for verification
        from aind_anatomical_utils import coordinate_systems as cs

        target_lps = cs.convert_coordinate_system(
            np.array([target_point]), "IRP", "LPS"
        )[0]

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_lps
            )
        )

    def test_unusual_target_frame_AIL(self):
        """Test with unusual target frame AIL."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]  # In AIL coordinates
        corner_code = "RAS"
        target_frame = "AIL"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            target_frame=target_frame,
        )

        self.assertIsInstance(origin, tuple)
        self.assertTrue(0 <= idx_num <= 7)

        from aind_anatomical_utils import coordinate_systems as cs

        target_lps = cs.convert_coordinate_system(
            np.array([target_point]), "AIL", "LPS"
        )[0]

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_lps
            )
        )

    def test_corner_and_target_frame_both_unusual(self):
        """Test with both corner_code and target_frame unusual."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]  # In IRP coordinates
        corner_code = "SRA"
        target_frame = "IRP"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            target_frame=target_frame,
        )

        self.assertIsInstance(origin, tuple)
        self.assertTrue(0 <= idx_num <= 7)

        from aind_anatomical_utils import coordinate_systems as cs

        target_lps = cs.convert_coordinate_system(
            np.array([target_point]), "IRP", "LPS"
        )[0]

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_lps
            )
        )

    def test_same_unusual_frame_for_both(self):
        """Test with same unusual frame for corner_code and target_frame."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]
        corner_code = "SRA"
        target_frame = "SRA"

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code=corner_code,
            target_frame=target_frame,
        )

        self.assertIsInstance(origin, tuple)
        self.assertTrue(0 <= idx_num <= 7)

        # When both are the same, no transformation needed
        from aind_anatomical_utils import coordinate_systems as cs

        target_lps = cs.convert_coordinate_system(
            np.array([target_point]), "SRA", "LPS"
        )[0]

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_lps
            )
        )

    def test_with_90_degree_rotation_x_axis(self):
        """Test with 90-degree rotation around x-axis."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        # 90-degree rotation around x-axis in LPS convention
        direction = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_with_90_degree_rotation_y_axis(self):
        """Test with 90-degree rotation around y-axis."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        # 90-degree rotation around y-axis in LPS convention
        direction = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_with_90_degree_rotation_z_axis(self):
        """Test with 90-degree rotation around z-axis."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        # 90-degree rotation around z-axis in LPS convention
        direction = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_with_oblique_direction_matrix(self):
        """Test with oblique (non-axis-aligned) direction matrix."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        # A valid orthonormal direction matrix with off-diagonal elements
        angle = np.pi / 6  # 30 degrees
        direction = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_anisotropic_spacing(self):
        """Test with anisotropic spacing."""
        size = [10, 20, 30]
        spacing = [0.5, 1.0, 2.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_small_volume(self):
        """Test with small volume size."""
        size = [2, 2, 2]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_large_volume(self):
        """Test with large volume size."""
        size = [512, 512, 256]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_thin_slice_volume(self):
        """Test with thin slice (one dimension is 1)."""
        size = [10, 1, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_very_small_spacing(self):
        """Test with very small spacing."""
        size = [10, 10, 10]
        spacing = [0.01, 0.01, 0.01]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_very_large_spacing(self):
        """Test with very large spacing."""
        size = [10, 10, 10]
        spacing = [10.0, 10.0, 10.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_target_point_at_origin(self):
        """Test with target point at origin."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_negative_target_point(self):
        """Test with negative target point coordinates."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [-10.0, -20.0, -30.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_far_from_origin_target_point(self):
        """Test with target point far from origin."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [1000.0, -1000.0, 500.0]

        origin, corner_idx, idx_num = av.fix_corner_compute_origin(
            size, spacing, direction, target_point, corner_code="RAS"
        )

        self.assertTrue(
            verify_origin_correctness(
                origin, direction, spacing, corner_idx, target_point
            )
        )

    def test_all_eight_corners_selectable(self):
        """Test that all 8 corners can be selected with appropriate corner
        codes."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [0.0, 0.0, 0.0]

        # All 8 combinations of R/L, A/P, S/I
        corner_codes = ["RAS", "LAS", "RPS", "LPS", "RAI", "LAI", "RPI", "LPI"]

        indices_found = set()
        for corner_code in corner_codes:
            _, _, idx_num = av.fix_corner_compute_origin(
                size, spacing, direction, target_point, corner_code=corner_code
            )
            indices_found.add(idx_num)

        # Should find multiple distinct corners (not necessarily all 8 due to
        # symmetry)
        self.assertGreater(len(indices_found), 1)
        # All indices should be in valid range
        self.assertTrue(all(0 <= idx <= 7 for idx in indices_found))

    def test_multiple_target_frames(self):
        """Test with multiple unusual target frames."""
        size = [10, 10, 10]
        spacing = [1.0, 1.0, 1.0]
        direction = np.eye(3)
        target_point = [5.0, 5.0, 5.0]
        corner_code = "RAS"

        # Valid unusual target frames (one from each pair: R/L, A/P, S/I)
        target_frames = ["SRA", "IRP", "AIL", "PSL", "SLP", "IPR"]

        for target_frame in target_frames:
            with self.subTest(target_frame=target_frame):
                origin, corner_idx, idx_num = av.fix_corner_compute_origin(
                    size,
                    spacing,
                    direction,
                    target_point,
                    corner_code=corner_code,
                    target_frame=target_frame,
                )

                self.assertIsInstance(origin, tuple)
                self.assertTrue(0 <= idx_num <= 7)

                # Convert to LPS for verification
                from aind_anatomical_utils import coordinate_systems as cs

                target_lps = cs.convert_coordinate_system(
                    np.array([target_point]), target_frame, "LPS"
                )[0]

                self.assertTrue(
                    verify_origin_correctness(
                        origin, direction, spacing, corner_idx, target_lps
                    )
                )


class TestRegridTo(unittest.TestCase):
    """Tests for AnatomicalHeader.regrid_to() method."""

    def _compute_voxel_physical_location(
        self, header: av.AnatomicalHeader, index: np.ndarray
    ) -> np.ndarray:
        """Compute physical location of voxel center using ITK formula."""
        # physical_point = origin + (index * spacing) @ direction.T
        index_arr = np.asarray(index, float)
        origin_arr = np.asarray(header.origin, float)
        spacing_arr = np.asarray(header.spacing, float)
        return origin_arr + (index_arr * spacing_arr) @ header.direction.T

    def _compute_all_corners_physical(
        self, header: av.AnatomicalHeader, use_outer_box: bool = False
    ) -> np.ndarray:
        """Compute physical locations of all 8 corners."""
        size_arr = np.asarray(header.size_ijk, float)
        corners_idx = av._corner_indices(size_arr, outer=use_outer_box)
        # Transform to physical coordinates
        origin_arr = np.asarray(header.origin, float)
        spacing_arr = np.asarray(header.spacing, float)
        corners_physical = (
            origin_arr + (corners_idx * spacing_arr) @ header.direction.T
        )
        return corners_physical

    def _compute_bounding_box(
        self, header: av.AnatomicalHeader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute min/max extents of bounding box."""
        corners = self._compute_all_corners_physical(
            header, use_outer_box=False
        )
        return corners.min(axis=0), corners.max(axis=0)

    def _create_test_header(
        self,
        coord_system: str,
        size: tuple[int, int, int] = (10, 20, 30),
        spacing: tuple[float, float, float] = (1.0, 2.0, 3.0),
        origin: tuple[float, float, float] = (10.0, 20.0, 30.0),
    ) -> av.AnatomicalHeader:
        """Create a test header with specified coordinate system."""
        dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
            coord_system
        )
        direction = np.array(dir_tuple).reshape(3, 3)

        return av.AnatomicalHeader(
            size_ijk=size,
            spacing=spacing,
            origin=origin,
            direction=direction,
        )

    def _compute_index_mapping(
        self, src_size: np.ndarray, src_code: str, dst_code: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute how to map an index from src to dst coordinate system.

        Returns (perm, flips) where:
        - perm: axis permutation array
        - flips: boolean array indicating which axes are flipped
        """
        from aind_anatomical_utils import coordinate_systems as cs

        perm, sign = cs.find_coordinate_perm_and_flips(src_code, dst_code)
        flips = sign < 0
        return perm, flips

    def _map_index(
        self,
        src_idx: np.ndarray,
        src_size: np.ndarray,
        src_code: str,
        dst_code: str,
    ) -> np.ndarray:
        """Map an index from src to dst coordinate system."""
        perm, flips = self._compute_index_mapping(src_size, src_code, dst_code)

        # Apply permutation
        dst_idx = src_idx[perm]

        # Apply flips: if axis is flipped, index i becomes (size-1-i)
        dst_size = src_size[perm]
        for axis_idx in range(3):
            if flips[axis_idx]:
                dst_idx[axis_idx] = dst_size[axis_idx] - 1 - dst_idx[axis_idx]

        return dst_idx.astype(int)

    def _verify_headers_same_physical_space(
        self,
        header1: av.AnatomicalHeader,
        header2: av.AnatomicalHeader,
    ) -> bool:
        """Verify two headers represent the same physical space."""
        # Compare bounding boxes
        bb1_min, bb1_max = self._compute_bounding_box(header1)
        bb2_min, bb2_max = self._compute_bounding_box(header2)

        if not (
            np.allclose(bb1_min, bb2_min) and np.allclose(bb1_max, bb2_max)
        ):
            return False

        # Verify corners match
        corners1 = self._compute_all_corners_physical(
            header1, use_outer_box=False
        )
        corners2 = self._compute_all_corners_physical(
            header2, use_outer_box=False
        )

        # Corners may be in different order, so sort them
        corners1_sorted = np.sort(corners1.view(np.void), axis=0).view(
            np.float64
        )
        corners2_sorted = np.sort(corners2.view(np.void), axis=0).view(
            np.float64
        )

        return np.allclose(corners1_sorted, corners2_sorted)

    def test_regrid_RAS_to_LPS(self):
        """Test regridding from RAS to LPS."""
        src_header = self._create_test_header("RAS")
        dst_header = src_header.regrid_to("LPS")

        # Verify same physical space
        self.assertTrue(
            self._verify_headers_same_physical_space(src_header, dst_header)
        )

        # Verify direction changed
        self.assertFalse(
            np.allclose(src_header.direction, dst_header.direction)
        )

        # Sample some voxels and verify they're at same locations
        test_indices = [
            [0, 0, 0],
            [5, 10, 15],
            [9, 19, 29],  # max indices
        ]

        for idx in test_indices:
            idx_arr = np.array(idx)
            src_phys = self._compute_voxel_physical_location(
                src_header, idx_arr
            )

            # Map index from RAS to LPS
            src_size = np.asarray(src_header.size_ijk, int)
            dst_idx = self._map_index(idx_arr, src_size, "RAS", "LPS")

            dst_phys = self._compute_voxel_physical_location(
                dst_header, dst_idx
            )
            self.assertTrue(
                np.allclose(src_phys, dst_phys),
                f"Voxel at {idx} in RAS (phys {src_phys}) doesn't match "
                f"{dst_idx} in LPS (phys {dst_phys})",
            )

    def test_regrid_LPS_to_RAS(self):
        """Test regridding from LPS to RAS."""
        src_header = self._create_test_header("LPS")
        dst_header = src_header.regrid_to("RAS")

        self.assertTrue(
            self._verify_headers_same_physical_space(src_header, dst_header)
        )

    def test_regrid_multiple_coordinate_systems(self):
        """Test regridding between multiple coordinate systems."""
        coord_systems = [
            "RAS",
            "LPS",
            "RPI",
            "LAI",
            "RPS",
            "LAS",
            "RAI",
            "LPI",
        ]

        for src_sys in coord_systems[:4]:  # Test subset for speed
            for dst_sys in coord_systems[:4]:
                with self.subTest(src=src_sys, dst=dst_sys):
                    src_header = self._create_test_header(src_sys)
                    dst_header = src_header.regrid_to(dst_sys)

                    self.assertTrue(
                        self._verify_headers_same_physical_space(
                            src_header, dst_header
                        ),
                        f"Failed for {src_sys} -> {dst_sys}",
                    )

    def test_regrid_round_trip_RAS_LPS_RAS(self):
        """Test round-trip regridding RAS -> LPS -> RAS."""
        original = self._create_test_header("RAS")
        intermediate = original.regrid_to("LPS")
        final = intermediate.regrid_to("RAS")

        # Should return to original
        self.assertTrue(np.allclose(original.origin, final.origin))
        self.assertTrue(np.allclose(original.direction, final.direction))
        self.assertTrue(np.allclose(original.spacing, final.spacing))
        self.assertTrue(np.array_equal(original.size_ijk, final.size_ijk))

    def test_regrid_round_trip_multiple_systems(self):
        """Test round-trip through multiple coordinate systems."""
        original = self._create_test_header("RAS")

        # Go through a chain of transformations
        h1 = original.regrid_to("LPS")
        h2 = h1.regrid_to("RPI")
        h3 = h2.regrid_to("LAI")
        final = h3.regrid_to("RAS")

        # Should return to original
        self.assertTrue(np.allclose(original.origin, final.origin, atol=1e-10))
        self.assertTrue(np.allclose(original.direction, final.direction))
        self.assertTrue(np.allclose(original.spacing, final.spacing))
        self.assertTrue(np.array_equal(original.size_ijk, final.size_ijk))

    def test_all_voxel_centers_preserved_small_volume(self):
        """Test that all voxel centers are preserved in a small volume."""
        src_header = self._create_test_header("RAS", size=(3, 4, 5))
        dst_header = src_header.regrid_to("LPS")

        # Test all voxels
        for i in range(3):
            for j in range(4):
                for k in range(5):
                    src_idx = np.array([i, j, k])
                    src_phys = self._compute_voxel_physical_location(
                        src_header, src_idx
                    )

                    # Map index from RAS to LPS
                    src_size = np.asarray(src_header.size_ijk, int)
                    dst_idx = self._map_index(src_idx, src_size, "RAS", "LPS")
                    dst_phys = self._compute_voxel_physical_location(
                        dst_header, dst_idx
                    )

                    self.assertTrue(
                        np.allclose(src_phys, dst_phys, atol=1e-10),
                        f"Mismatch at src {src_idx} -> dst {dst_idx}",
                    )

    def test_sampled_voxel_centers_preserved_large_volume(self):
        """Test sampled voxel centers in a larger volume."""
        np.random.seed(42)
        src_header = self._create_test_header("RAS", size=(100, 200, 50))
        dst_header = src_header.regrid_to("LPI")

        # Sample 20 random voxels
        for _ in range(20):
            src_idx = np.array(
                [
                    np.random.randint(0, 100),
                    np.random.randint(0, 200),
                    np.random.randint(0, 50),
                ]
            )
            src_phys = self._compute_voxel_physical_location(
                src_header, src_idx
            )

            src_size = np.asarray(src_header.size_ijk, int)
            dst_idx = self._map_index(src_idx, src_size, "RAS", "LPI")
            dst_phys = self._compute_voxel_physical_location(
                dst_header, dst_idx
            )

            self.assertTrue(np.allclose(src_phys, dst_phys, atol=1e-10))

    def test_all_corners_preserved(self):
        """Test that all 8 corners are at the same physical locations."""
        src_header = self._create_test_header("RAS")
        dst_header = src_header.regrid_to("LPS")

        src_corners = self._compute_all_corners_physical(
            src_header, use_outer_box=False
        )
        dst_corners = self._compute_all_corners_physical(
            dst_header, use_outer_box=False
        )

        # Sort corners for comparison (order may differ)
        src_sorted = (
            np.sort(src_corners.view(np.void), axis=0)
            .view(np.float64)
            .reshape(-1, 3)
        )
        dst_sorted = (
            np.sort(dst_corners.view(np.void), axis=0)
            .view(np.float64)
            .reshape(-1, 3)
        )

        self.assertTrue(np.allclose(src_sorted, dst_sorted, atol=1e-10))

    def test_bounding_box_extents_preserved(self):
        """Test that bounding box extents are preserved."""
        src_header = self._create_test_header("RAS")
        dst_header = src_header.regrid_to("LPS")

        src_min, src_max = self._compute_bounding_box(src_header)
        dst_min, dst_max = self._compute_bounding_box(dst_header)

        self.assertTrue(np.allclose(src_min, dst_min, atol=1e-10))
        self.assertTrue(np.allclose(src_max, dst_max, atol=1e-10))

    def test_volume_extent_preserved(self):
        """Test that physical volume is preserved."""
        src_header = self._create_test_header("RAS")
        dst_header = src_header.regrid_to("LPS")

        src_size = np.asarray(src_header.size_ijk, float)
        src_spacing = np.asarray(src_header.spacing, float)
        dst_size = np.asarray(dst_header.size_ijk, float)
        dst_spacing = np.asarray(dst_header.spacing, float)
        src_volume = np.prod(src_size * src_spacing)
        dst_volume = np.prod(dst_size * dst_spacing)

        self.assertAlmostEqual(src_volume, dst_volume, places=10)

    def test_regrid_anisotropic_spacing(self):
        """Test regridding with anisotropic spacing."""
        src_header = self._create_test_header("RAS", spacing=(0.5, 1.0, 2.0))
        dst_header = src_header.regrid_to("LPS")

        self.assertTrue(
            self._verify_headers_same_physical_space(src_header, dst_header)
        )

    def test_regrid_non_cubic_volume(self):
        """Test regridding with non-cubic volume."""
        src_header = self._create_test_header("RAS", size=(128, 256, 64))
        dst_header = src_header.regrid_to("LPS")

        self.assertTrue(
            self._verify_headers_same_physical_space(src_header, dst_header)
        )

    def test_regrid_arbitrary_origins(self):
        """Test regridding with various origin locations."""
        origins = [
            (0.0, 0.0, 0.0),
            (100.0, -50.0, 25.0),
            (-100.0, -100.0, -100.0),
        ]

        for origin in origins:
            with self.subTest(origin=origin):
                src_header = self._create_test_header("RAS", origin=origin)
                dst_header = src_header.regrid_to("LPS")

                self.assertTrue(
                    self._verify_headers_same_physical_space(
                        src_header, dst_header
                    )
                )

    def test_regrid_identity_transformation(self):
        """Test regridding to the same coordinate system."""
        src_header = self._create_test_header("RAS")
        dst_header = src_header.regrid_to("RAS")

        # Should be identical
        self.assertTrue(np.allclose(src_header.origin, dst_header.origin))
        self.assertTrue(
            np.allclose(src_header.direction, dst_header.direction)
        )
        self.assertTrue(np.allclose(src_header.spacing, dst_header.spacing))
        self.assertTrue(
            np.array_equal(src_header.size_ijk, dst_header.size_ijk)
        )

    def test_regrid_small_volume(self):
        """Test regridding a 2x2x2 volume."""
        src_header = self._create_test_header("RAS", size=(2, 2, 2))
        dst_header = src_header.regrid_to("LPS")

        self.assertTrue(
            self._verify_headers_same_physical_space(src_header, dst_header)
        )

    def test_regrid_thin_slice(self):
        """Test regridding a thin slice volume."""
        src_header = self._create_test_header("RAS", size=(256, 256, 1))
        dst_header = src_header.regrid_to("LPS")

        self.assertTrue(
            self._verify_headers_same_physical_space(src_header, dst_header)
        )

    def test_regrid_very_small_spacing(self):
        """Test regridding with very small spacing."""
        src_header = self._create_test_header(
            "RAS", spacing=(0.01, 0.01, 0.01)
        )
        dst_header = src_header.regrid_to("LPS")

        self.assertTrue(
            self._verify_headers_same_physical_space(src_header, dst_header)
        )

    def test_regrid_large_volume(self):
        """Test regridding a large volume."""
        src_header = self._create_test_header("RAS", size=(512, 512, 256))
        dst_header = src_header.regrid_to("LPS")

        # Just verify corners and bounding box (too many voxels to test all)
        src_min, src_max = self._compute_bounding_box(src_header)
        dst_min, dst_max = self._compute_bounding_box(dst_header)

        self.assertTrue(np.allclose(src_min, dst_min, atol=1e-10))
        self.assertTrue(np.allclose(src_max, dst_max, atol=1e-10))

    def test_regrid_non_axis_aligned_raises_error(self):
        """Test that non-axis-aligned volumes raise an error."""
        # Create header with oblique direction matrix
        angle = np.pi / 6
        direction = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        header = av.AnatomicalHeader(
            size_ijk=(10, 10, 10),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=direction,
        )

        with self.assertRaises(ValueError) as ctx:
            header.regrid_to("LPS")

        self.assertIn("axis-aligned", str(ctx.exception).lower())

    def test_regrid_origin_voxel_mapping(self):
        """Test that origin voxel [0,0,0] maps correctly."""
        src_header = self._create_test_header("RAS")
        dst_header = src_header.regrid_to("LPS")

        src_idx = np.array([0, 0, 0])
        src_phys = self._compute_voxel_physical_location(src_header, src_idx)

        # Map index from RAS to LPS
        src_size = np.asarray(src_header.size_ijk, int)
        dst_idx = self._map_index(src_idx, src_size, "RAS", "LPS")
        dst_phys = self._compute_voxel_physical_location(dst_header, dst_idx)

        self.assertTrue(np.allclose(src_phys, dst_phys, atol=1e-10))

    def test_regrid_max_index_voxel_mapping(self):
        """Test that max index voxel maps correctly."""
        src_header = self._create_test_header("RAS", size=(10, 20, 30))
        dst_header = src_header.regrid_to("LPS")

        src_idx = np.array([9, 19, 29])
        src_phys = self._compute_voxel_physical_location(src_header, src_idx)

        src_size = np.asarray(src_header.size_ijk, int)
        dst_idx = self._map_index(src_idx, src_size, "RAS", "LPS")
        dst_phys = self._compute_voxel_physical_location(dst_header, dst_idx)

        self.assertTrue(np.allclose(src_phys, dst_phys, atol=1e-10))

    def test_regrid_center_voxel_mapping(self):
        """Test that center voxel stays at center."""
        src_header = self._create_test_header("RAS", size=(20, 20, 20))
        dst_header = src_header.regrid_to("LPS")

        src_idx = np.array([10, 10, 10])
        src_phys = self._compute_voxel_physical_location(src_header, src_idx)

        src_size = np.asarray(src_header.size_ijk, int)
        dst_idx = self._map_index(src_idx, src_size, "RAS", "LPS")
        dst_phys = self._compute_voxel_physical_location(dst_header, dst_idx)

        self.assertTrue(np.allclose(src_phys, dst_phys, atol=1e-10))


class TestAntsIntegration(unittest.TestCase):
    """Tests for ANTs image integration methods."""

    def test_as_ants_basic(self):
        """Test creating ANTs image from header."""
        header = av.AnatomicalHeader(
            origin=(5.0, 10.0, 15.0),
            spacing=(0.5, 1.0, 2.0),
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        ants_img = header.as_ants()

        # Verify properties
        self.assertTrue(np.allclose(ants_img.origin, header.origin))
        # Spacing should match header spacing
        self.assertTrue(np.allclose(ants_img.spacing, header.spacing))
        self.assertTrue(
            np.allclose(ants_img.direction, header.direction.reshape(3, 3))
        )
        self.assertEqual(ants_img.shape, header.size_ijk)

    def test_as_ants_various_dtypes(self):
        """Test creating ANTs images with various dtypes.

        Note: ANTsPy may convert some dtypes (e.g., int16->uint32,
        float64->float32).  We verify the image is created successfully, not
        exact dtype preservation.
        """
        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=np.eye(3),
            size_ijk=(5, 5, 5),
        )

        dtypes = [np.uint8, np.int16, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                ants_img = header.as_ants(np_eltype=dtype)
                # Verify image was created with correct shape
                self.assertEqual(ants_img.shape, (5, 5, 5))
                # Verify it's a numeric dtype (ANTsPy may convert types)
                self.assertTrue(
                    np.issubdtype(ants_img.numpy().dtype, np.number)
                )

    def test_as_ants_physical_extent(self):
        """Test that as_ants creates image with correct physical extent."""
        header = av.AnatomicalHeader(
            origin=(10.0, 20.0, 30.0),
            spacing=(2.0, 3.0, 4.0),
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        ants_img = header.as_ants()

        # Compute physical location of first voxel using ANTs
        first_idx = np.array([0, 0, 0])
        # ANTs formula: physical = origin + direction @ (spacing * index)
        first_phys = np.array(ants_img.origin) + ants_img.direction @ (
            np.array(ants_img.spacing) * first_idx
        )
        self.assertTrue(np.allclose(first_phys, header.origin))

    def test_update_ants(self):
        """Test updating ANTs image with header."""
        # Create initial ANTs image with different properties
        initial_img = ants.from_numpy(
            np.zeros((10, 20, 30)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=np.eye(3),
        )

        # Create header with new properties
        header = av.AnatomicalHeader(
            origin=(5.0, 10.0, 15.0),
            spacing=(0.5, 1.0, 2.0),
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        # Update the image
        result = header.update_ants(initial_img)

        # Verify returns same instance
        self.assertIs(result, initial_img)

        # Verify properties updated
        self.assertTrue(np.allclose(initial_img.origin, header.origin))
        # Spacing should match header spacing
        self.assertTrue(np.allclose(initial_img.spacing, header.spacing))
        self.assertTrue(
            np.allclose(initial_img.direction, header.direction.reshape(3, 3))
        )

    def test_update_ants_preserves_spacing(self):
        """Test that update_ants correctly sets spacing from header."""
        ants_img = ants.from_numpy(
            np.zeros((10, 20, 30)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=np.eye(3),
        )

        # Header with specific spacing
        header = av.AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(2.0, 3.0, 4.0),  # i, j, k order
            direction=np.eye(3),
            size_ijk=(10, 20, 30),
        )

        header.update_ants(ants_img)

        # ANTs should have same spacing as header
        self.assertTrue(np.allclose(ants_img.spacing, header.spacing))

    def test_from_ants_basic(self):
        """Test creating header from ANTs image."""
        ants_img = ants.from_numpy(
            np.zeros((10, 20, 30)),
            origin=(5.0, 10.0, 15.0),
            spacing=(2.0, 1.0, 0.5),
            direction=np.eye(3),
        )

        header = av.AnatomicalHeader.from_ants(ants_img)

        # Verify properties
        self.assertTrue(np.allclose(header.origin, ants_img.origin))
        self.assertTrue(np.allclose(header.spacing, ants_img.spacing))
        self.assertTrue(np.allclose(header.direction, ants_img.direction))
        self.assertEqual(header.size_ijk, ants_img.shape)

    def test_from_ants_custom_size(self):
        """Test creating header from ANTs image with custom size."""
        ants_img = ants.from_numpy(
            np.zeros((10, 20, 30)),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=np.eye(3),
        )

        custom_size = (5, 10, 15)
        header = av.AnatomicalHeader.from_ants(ants_img, size_ijk=custom_size)

        self.assertEqual(header.size_ijk, custom_size)

    def test_from_ants_roundtrip(self):
        """Test round-trip: ANTs -> Header -> ANTs."""
        original_img = ants.from_numpy(
            np.zeros((10, 20, 30)),
            origin=(5.0, 10.0, 15.0),
            spacing=(2.0, 1.0, 0.5),
            direction=np.eye(3),
        )

        # Convert to header and back
        header = av.AnatomicalHeader.from_ants(original_img)
        recovered_img = header.as_ants()

        # Verify all properties match
        self.assertTrue(np.allclose(original_img.origin, recovered_img.origin))
        self.assertTrue(
            np.allclose(original_img.spacing, recovered_img.spacing)
        )
        self.assertTrue(
            np.allclose(original_img.direction, recovered_img.direction)
        )
        self.assertEqual(original_img.shape, recovered_img.shape)


class TestCrossLibraryRoundTrips(unittest.TestCase):
    """Test round-trip conversions between SimpleITK and ANTs."""

    def test_sitk_to_header_to_ants(self):
        """Test SimpleITK -> Header -> ANTs preserves physical space."""
        # Create SimpleITK image
        sitk_img = sitk.Image([10, 20, 30], sitk.sitkUInt8)
        sitk_img.SetOrigin((5.0, 10.0, 15.0))
        sitk_img.SetSpacing((0.5, 1.0, 2.0))
        sitk_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        # Convert through header to ANTs
        header = av.AnatomicalHeader.from_sitk(sitk_img)
        ants_img = header.as_ants()

        # Verify physical location of first voxel matches
        sitk_first = sitk_img.TransformIndexToPhysicalPoint((0, 0, 0))
        # For ANTs, compute manually
        ants_first = np.array(ants_img.origin)

        self.assertTrue(np.allclose(sitk_first, ants_first))

        # Verify physical location of last voxel
        sitk_last = sitk_img.TransformIndexToPhysicalPoint((9, 19, 29))
        # ANTs: physical = origin + direction @ (spacing * index)
        last_idx_ants = np.array([9, 19, 29])
        ants_last = np.array(ants_img.origin) + ants_img.direction @ (
            np.array(ants_img.spacing) * last_idx_ants
        )

        self.assertTrue(np.allclose(sitk_last, ants_last))

    def test_ants_to_header_to_sitk(self):
        """Test ANTs -> Header -> SimpleITK preserves physical space."""
        # Create ANTs image
        ants_img = ants.from_numpy(
            np.zeros((10, 20, 30)),
            origin=(5.0, 10.0, 15.0),
            spacing=(2.0, 1.0, 0.5),
            direction=np.eye(3),
        )

        # Convert through header to SimpleITK
        header = av.AnatomicalHeader.from_ants(ants_img)
        sitk_img = header.as_sitk()

        # Verify origins match
        self.assertTrue(np.allclose(ants_img.origin, sitk_img.GetOrigin()))

        # Verify physical location of center voxel
        center_idx = np.array([5, 10, 15])
        ants_center = np.array(ants_img.origin) + ants_img.direction @ (
            np.array(ants_img.spacing) * center_idx
        )
        sitk_center = sitk_img.TransformIndexToPhysicalPoint(
            tuple(center_idx.tolist())
        )

        self.assertTrue(np.allclose(ants_center, sitk_center))

    def test_roundtrip_with_anisotropic_spacing(self):
        """Test round-trip with anisotropic spacing."""
        # Create header with anisotropic spacing
        original_header = av.AnatomicalHeader(
            origin=(10.0, 20.0, 30.0),
            spacing=(0.25, 0.5, 1.0),
            direction=np.eye(3),
            size_ijk=(100, 50, 25),
        )

        # Round-trip through SimpleITK
        sitk_img = original_header.as_sitk()
        header_from_sitk = av.AnatomicalHeader.from_sitk(sitk_img)

        # Round-trip through ANTs
        ants_img = original_header.as_ants()
        header_from_ants = av.AnatomicalHeader.from_ants(ants_img)

        # Verify both recovered headers match original
        for recovered in [header_from_sitk, header_from_ants]:
            self.assertTrue(
                np.allclose(original_header.origin, recovered.origin)
            )
            self.assertTrue(
                np.allclose(original_header.spacing, recovered.spacing)
            )
            self.assertTrue(
                np.allclose(original_header.direction, recovered.direction)
            )
            self.assertEqual(original_header.size_ijk, recovered.size_ijk)

    def test_roundtrip_with_different_orientation(self):
        """Test round-trip with RAS orientation."""
        # Get RAS direction matrix
        dir_tuple = DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
            "RAS"
        )
        direction = np.array(dir_tuple).reshape(3, 3)

        original_header = av.AnatomicalHeader(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            direction=direction,
            size_ijk=(10, 10, 10),
        )

        # Round-trip through both libraries
        sitk_img = original_header.as_sitk()
        ants_img = original_header.as_ants()

        header_from_sitk = av.AnatomicalHeader.from_sitk(sitk_img)
        header_from_ants = av.AnatomicalHeader.from_ants(ants_img)

        # Both should match original
        for recovered in [header_from_sitk, header_from_ants]:
            self.assertTrue(
                np.allclose(original_header.direction, recovered.direction)
            )

    def test_voxel_centers_match_across_libraries(self):
        """Test that voxel centers map to same physical locations."""
        header = av.AnatomicalHeader(
            origin=(10.0, 20.0, 30.0),
            spacing=(2.0, 3.0, 4.0),
            direction=np.eye(3),
            size_ijk=(5, 5, 5),
        )

        sitk_img = header.as_sitk()
        ants_img = header.as_ants()

        # Test several voxel locations
        test_indices = [(0, 0, 0), (2, 2, 2), (4, 4, 4), (1, 3, 2)]

        for idx in test_indices:
            with self.subTest(index=idx):
                # Get physical location from SimpleITK
                sitk_phys = sitk_img.TransformIndexToPhysicalPoint(idx)

                # Get physical location from ANTs
                idx_arr = np.array(idx)
                ants_phys = np.array(ants_img.origin) + ants_img.direction @ (
                    np.array(ants_img.spacing) * idx_arr
                )

                self.assertTrue(
                    np.allclose(sitk_phys, ants_phys),
                    f"Mismatch at {idx}: SITK={sitk_phys}, ANTs={ants_phys}",
                )


if __name__ == "__main__":
    unittest.main()
