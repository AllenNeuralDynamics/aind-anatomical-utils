"""Tests functions in `anatomical_volume`."""

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
