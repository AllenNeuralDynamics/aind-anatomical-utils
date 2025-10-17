"""Tests for ants_volume module."""

import unittest

import ants  # type: ignore[import-untyped]
import numpy as np

from aind_anatomical_utils import ants_volume

# Import shared helpers from conftest (pytest auto-discovers conftest.py)
from .conftest import (
    create_gradient_ants_image,
    create_gradient_sitk_image,
    get_ants_orientation_code,
    get_ants_voxel_value_at_physical_point,
    get_sitk_voxel_value_at_physical_point,
)


class TestRegridAxisAlignedAnts(unittest.TestCase):
    """Tests for regrid_axis_aligned_ants function."""

    def test_regrid_RAS_to_LPS(self):
        """Test regridding from RAS to LPS with value verification."""
        img = create_gradient_ants_image("RAS", size=(5, 6, 7))
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Verify output orientation
        result_code = get_ants_orientation_code(result)
        self.assertEqual(result_code, "LPS")

        # Verify voxel values at physical locations
        # Test corner voxel [0,0,0] in RAS
        src_phys = np.array(img.origin) + img.direction @ (
            np.array(img.spacing) * np.array([0, 0, 0])
        )
        src_val = img.numpy()[0, 0, 0]
        result_val = get_ants_voxel_value_at_physical_point(
            result, tuple(src_phys)
        )
        self.assertEqual(src_val, result_val)

    def test_regrid_LPS_to_RAS(self):
        """Test regridding from LPS to RAS."""
        img = create_gradient_ants_image("LPS", size=(5, 6, 7))
        result = ants_volume.regrid_axis_aligned_ants(img, "RAS")

        # Verify output orientation
        result_code = get_ants_orientation_code(result)
        self.assertEqual(result_code, "RAS")

        # Verify values preserved at physical locations
        src_phys = np.array(img.origin) + img.direction @ (
            np.array(img.spacing) * np.array([2, 3, 4])
        )
        src_val = img.numpy()[2, 3, 4]
        result_val = get_ants_voxel_value_at_physical_point(
            result, tuple(src_phys)
        )
        self.assertEqual(src_val, result_val)

    def test_regrid_to_multiple_orientations(self):
        """Test regridding to various orientations."""
        orientations = ["RAS", "LPS", "RPI", "LAI"]
        src_img = create_gradient_ants_image("RAS", size=(4, 5, 6))

        for dst_orient in orientations:
            with self.subTest(dst_orient=dst_orient):
                result = ants_volume.regrid_axis_aligned_ants(
                    src_img, dst_orient
                )

                # Verify correct orientation
                result_code = get_ants_orientation_code(result)
                self.assertEqual(result_code, dst_orient)

                # Verify at least one voxel value preserved
                src_phys = np.array(src_img.origin) + src_img.direction @ (
                    np.array(src_img.spacing) * np.array([1, 1, 1])
                )
                src_val = src_img.numpy()[1, 1, 1]
                result_val = get_ants_voxel_value_at_physical_point(
                    result, tuple(src_phys)
                )
                self.assertEqual(src_val, result_val)

    def test_regrid_identity_transformation(self):
        """Test regridding to the same coordinate system."""
        img = create_gradient_ants_image("RAS", size=(5, 6, 7))
        result = ants_volume.regrid_axis_aligned_ants(img, "RAS")

        # Should be identical
        self.assertEqual(img.shape, result.shape)
        self.assertTrue(np.allclose(img.origin, result.origin))
        self.assertTrue(np.allclose(img.spacing, result.spacing))
        self.assertTrue(np.allclose(img.direction, result.direction))

        # All voxel values should match
        self.assertTrue(np.array_equal(img.numpy(), result.numpy()))

    def test_regrid_round_trip_preserves_values(self):
        """Test RAS→LPS→RAS round trip preserves data."""
        original = create_gradient_ants_image("RAS", size=(5, 6, 7))
        intermediate = ants_volume.regrid_axis_aligned_ants(original, "LPS")
        final = ants_volume.regrid_axis_aligned_ants(intermediate, "RAS")

        # Should return to original state
        self.assertEqual(original.shape, final.shape)
        self.assertTrue(np.allclose(original.origin, final.origin, atol=1e-10))
        self.assertTrue(
            np.allclose(original.spacing, final.spacing, atol=1e-10)
        )
        self.assertTrue(
            np.allclose(original.direction, final.direction, atol=1e-10)
        )

        # All voxel values should match
        self.assertTrue(np.array_equal(original.numpy(), final.numpy()))

    def test_corner_voxels_preserved_at_physical_locations(self):
        """Test that corner voxel values are at correct physical locations."""
        img = create_gradient_ants_image("RAS", size=(5, 6, 7))
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Test all 8 corners
        size = img.shape
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
            src_phys = np.array(img.origin) + img.direction @ (
                np.array(img.spacing) * np.array(corner)
            )
            src_val = img.numpy()[corner[0], corner[1], corner[2]]
            result_val = get_ants_voxel_value_at_physical_point(
                result, tuple(src_phys)
            )
            self.assertEqual(
                src_val,
                result_val,
                f"Corner {corner} mismatch: {src_val} != {result_val}",
            )

    def test_center_voxel_preserved(self):
        """Test that center voxel value remains at center after regridding."""
        img = create_gradient_ants_image("RAS", size=(10, 20, 30))
        result = ants_volume.regrid_axis_aligned_ants(img, "LPI")

        # Test center voxel
        center_idx = (5, 10, 15)
        src_phys = np.array(img.origin) + img.direction @ (
            np.array(img.spacing) * np.array(center_idx)
        )
        src_val = img.numpy()[center_idx[0], center_idx[1], center_idx[2]]
        result_val = get_ants_voxel_value_at_physical_point(
            result, tuple(src_phys)
        )
        self.assertEqual(src_val, result_val)

    def test_sampled_voxels_preserved(self):
        """Test that sampled voxel values are preserved at physical
        locations."""
        np.random.seed(42)
        img = create_gradient_ants_image("RAS", size=(10, 15, 20))
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Sample 10 random voxels
        for _ in range(10):
            idx = (
                np.random.randint(0, 10),
                np.random.randint(0, 15),
                np.random.randint(0, 20),
            )
            src_phys = np.array(img.origin) + img.direction @ (
                np.array(img.spacing) * np.array(idx)
            )
            src_val = img.numpy()[idx[0], idx[1], idx[2]]
            result_val = get_ants_voxel_value_at_physical_point(
                result, tuple(src_phys)
            )
            self.assertEqual(src_val, result_val)

    def test_regrid_with_uint8(self):
        """Test regridding with uint8 dtype."""
        arr = np.random.randint(0, 256, size=(5, 6, 7), dtype=np.uint8)
        img = ants.from_numpy(arr, origin=(0, 0, 0), spacing=(1, 1, 1))

        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Should preserve dtype
        self.assertEqual(result.numpy().dtype, np.uint8)
        self.assertEqual(result.shape, (5, 6, 7))

    def test_regrid_with_float32(self):
        """Test regridding with float32 dtype."""
        arr = np.random.randn(5, 6, 7).astype(np.float32)
        img = ants.from_numpy(arr, origin=(0, 0, 0), spacing=(1, 1, 1))

        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Should preserve dtype (ANTs may convert to float32)
        self.assertTrue(np.issubdtype(result.numpy().dtype, np.floating))
        self.assertEqual(result.shape, (5, 6, 7))

    def test_regrid_with_anisotropic_spacing(self):
        """Test regridding with anisotropic spacing."""
        img = create_gradient_ants_image(
            "RAS", size=(5, 6, 7), spacing=(0.5, 1.0, 2.0)
        )
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Verify orientation changed
        result_code = get_ants_orientation_code(result)
        self.assertEqual(result_code, "LPS")

        # Verify corner preserved
        src_phys = np.array(img.origin) + img.direction @ (
            np.array(img.spacing) * np.array([0, 0, 0])
        )
        src_val = img.numpy()[0, 0, 0]
        result_val = get_ants_voxel_value_at_physical_point(
            result, tuple(src_phys)
        )
        self.assertEqual(src_val, result_val)

    def test_regrid_with_non_zero_origin(self):
        """Test regridding with non-zero origin."""
        img = create_gradient_ants_image(
            "RAS", size=(5, 6, 7), origin=(100.0, -50.0, 25.0)
        )
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Verify orientation changed
        result_code = get_ants_orientation_code(result)
        self.assertEqual(result_code, "LPS")

        # Verify center voxel preserved
        center_idx = (2, 3, 3)
        src_phys = np.array(img.origin) + img.direction @ (
            np.array(img.spacing) * np.array(center_idx)
        )
        src_val = img.numpy()[center_idx[0], center_idx[1], center_idx[2]]
        result_val = get_ants_voxel_value_at_physical_point(
            result, tuple(src_phys)
        )
        self.assertEqual(src_val, result_val)

    def test_regrid_small_volume(self):
        """Test regridding a 2x2x2 volume."""
        img = create_gradient_ants_image("RAS", size=(2, 2, 2))
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        result_code = get_ants_orientation_code(result)
        self.assertEqual(result_code, "LPS")
        self.assertEqual(result.shape, (2, 2, 2))

        # Verify all 8 voxels preserved
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    src_phys = np.array(img.origin) + img.direction @ (
                        np.array(img.spacing) * np.array([i, j, k])
                    )
                    src_val = img.numpy()[i, j, k]
                    result_val = get_ants_voxel_value_at_physical_point(
                        result, tuple(src_phys)
                    )
                    self.assertEqual(src_val, result_val)

    def test_regrid_thin_slice(self):
        """Test regridding a thin slice volume."""
        img = create_gradient_ants_image("RAS", size=(256, 256, 1))
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        result_code = get_ants_orientation_code(result)
        self.assertEqual(result_code, "LPS")

        # Verify corner preserved
        src_phys = np.array(img.origin)
        src_val = img.numpy()[0, 0, 0]
        result_val = get_ants_voxel_value_at_physical_point(
            result, tuple(src_phys)
        )
        self.assertEqual(src_val, result_val)

    def test_regrid_non_axis_aligned_raises_error(self):
        """Test that non-axis-aligned volumes raise an error."""
        # Create image with oblique direction matrix
        arr = np.zeros((5, 6, 7), dtype=np.int32)
        angle = np.pi / 6  # 30 degrees
        direction = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        img = ants.from_numpy(
            arr, origin=(0, 0, 0), spacing=(1, 1, 1), direction=direction
        )

        with self.assertRaises(ValueError) as ctx:
            ants_volume.regrid_axis_aligned_ants(img, "LPS")

        self.assertIn("axis-aligned", str(ctx.exception).lower())

    def test_regrid_labeled_volume_preserves_labels(self):
        """Test that labeled volume preserves discrete label values."""
        # Create image with discrete labels
        arr = np.zeros((5, 6, 7), dtype=np.int32)
        arr[1:3, 2:4, 3:5] = 1  # Region 1
        arr[3:5, 4:6, 1:3] = 2  # Region 2

        img = ants.from_numpy(arr, origin=(0, 0, 0), spacing=(1, 1, 1))
        result = ants_volume.regrid_axis_aligned_ants(img, "LPS")

        # Verify each labeled voxel is preserved at physical location
        for i in range(5):
            for j in range(6):
                for k in range(7):
                    if arr[i, j, k] > 0:
                        src_phys = np.array(img.origin) + img.direction @ (
                            np.array(img.spacing) * np.array([i, j, k])
                        )
                        src_val = arr[i, j, k]
                        result_val = get_ants_voxel_value_at_physical_point(
                            result, tuple(src_phys)
                        )
                        self.assertEqual(src_val, result_val)


class TestCrossLibraryValidation(unittest.TestCase):
    """Tests that verify ANTs and SimpleITK produce equivalent results."""

    def test_both_libraries_produce_same_corner_locations(self):
        """Test that both regrid functions produce same physical corners."""
        from aind_anatomical_utils import sitk_volume

        # Create identical gradient images in both libraries
        sitk_img = create_gradient_sitk_image("RAS", size=(5, 6, 7))
        ants_img = create_gradient_ants_image("RAS", size=(5, 6, 7))

        # Regrid both to LPS
        sitk_result = sitk_volume.regrid_axis_aligned_sitk(sitk_img, "LPS")
        ants_result = ants_volume.regrid_axis_aligned_ants(ants_img, "LPS")

        # Test all 8 corner locations match
        size = sitk_img.GetSize()
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
            # Get SimpleITK physical location
            sitk_phys = sitk_result.TransformIndexToPhysicalPoint(corner)

            # Get ANTs physical location
            ants_phys = np.array(
                ants_result.origin
            ) + ants_result.direction @ (
                np.array(ants_result.spacing) * np.array(corner)
            )

            self.assertTrue(
                np.allclose(sitk_phys, ants_phys, atol=1e-10),
                (
                    f"Corner {corner} mismatch: "
                    f"SITK={sitk_phys}, ANTs={ants_phys}"
                ),
            )

    def test_gradient_values_match_across_libraries(self):
        """Test that gradient values match at same physical locations."""
        from aind_anatomical_utils import sitk_volume

        # Create identical gradient images
        sitk_img = create_gradient_sitk_image("RAS", size=(5, 6, 7))
        ants_img = create_gradient_ants_image("RAS", size=(5, 6, 7))

        # Regrid both to same orientation
        sitk_result = sitk_volume.regrid_axis_aligned_sitk(sitk_img, "LPI")
        ants_result = ants_volume.regrid_axis_aligned_ants(ants_img, "LPI")

        # Sample several voxels and verify values match
        test_indices = [(0, 0, 0), (2, 3, 4), (4, 5, 6), (1, 2, 3)]

        for idx in test_indices:
            # Get physical location from SimpleITK
            sitk_phys = sitk_img.TransformIndexToPhysicalPoint(idx)

            # Get values at that physical location in both results
            sitk_val = get_sitk_voxel_value_at_physical_point(
                sitk_result, sitk_phys
            )
            ants_val = get_ants_voxel_value_at_physical_point(
                ants_result, tuple(sitk_phys)
            )

            self.assertEqual(
                sitk_val,
                ants_val,
                f"Value mismatch at {idx}: SITK={sitk_val}, ANTs={ants_val}",
            )

    def test_both_regrid_functions_preserve_physical_bounding_box(self):
        """Test that both libraries preserve the physical bounding box."""
        from aind_anatomical_utils import sitk_volume

        # Create identical gradient images
        sitk_img = create_gradient_sitk_image(
            "RAS", size=(5, 6, 7), spacing=(1.0, 2.0, 3.0)
        )
        ants_img = create_gradient_ants_image(
            "RAS", size=(5, 6, 7), spacing=(1.0, 2.0, 3.0)
        )

        # Regrid both to LPS
        sitk_result = sitk_volume.regrid_axis_aligned_sitk(sitk_img, "LPS")
        ants_result = ants_volume.regrid_axis_aligned_ants(ants_img, "LPS")

        # Get bounding boxes (min and max corners in physical space)
        # For SimpleITK
        sitk_min_phys = sitk_result.TransformIndexToPhysicalPoint((0, 0, 0))
        sitk_size = sitk_result.GetSize()
        sitk_max_idx = tuple(s - 1 for s in sitk_size)
        sitk_max_phys = sitk_result.TransformIndexToPhysicalPoint(sitk_max_idx)

        # For ANTs
        ants_min_phys = np.array(ants_result.origin)
        ants_size = ants_result.shape
        ants_max_idx = np.array([s - 1 for s in ants_size])
        ants_max_phys = ants_min_phys + ants_result.direction @ (
            np.array(ants_result.spacing) * ants_max_idx
        )

        # Bounding boxes should match
        self.assertTrue(
            np.allclose(sitk_min_phys, ants_min_phys, atol=1e-10),
            f"Min corner mismatch: SITK={sitk_min_phys}, ANTs={ants_min_phys}",
        )
        self.assertTrue(
            np.allclose(sitk_max_phys, ants_max_phys, atol=1e-10),
            f"Max corner mismatch: SITK={sitk_max_phys}, ANTs={ants_max_phys}",
        )


if __name__ == "__main__":
    unittest.main()
