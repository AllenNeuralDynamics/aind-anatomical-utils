import unittest

import numpy as np
import SimpleITK as sitk

from aind_anatomical_utils import sitk_volume


def all_closer_than(a, b, thresh):
    return np.all(np.abs(a - b) <= thresh)


def fraction_close(a, val):
    arr = sitk.GetArrayViewFromImage(a)
    nel = np.prod(arr.shape)
    return np.sum(np.isclose(arr, val)) / nel


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
        for ndxs, ans in self.test_index_translation_sets:
            received = sitk_volume.transform_sitk_indices_to_physical_points(
                simg, ndxs
            )
            self.assertTrue(np.allclose(ans, received))


if __name__ == "__main__":
    unittest.main()
