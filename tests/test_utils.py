"""Tests functions in `utils`."""

import unittest

import numpy as np

from aind_anatomical_utils.utils import find_indices_equal_to


class UtilsTest(unittest.TestCase):
    """Tests functions in `utils`."""

    test_match_arr = np.arange(0, 9).reshape(3, 3)

    def test_find_indices_equal_to(self) -> None:
        self.assertTrue(
            np.array_equal(
                find_indices_equal_to(self.test_match_arr, 1),
                np.array([[0, 1]]),
            )
        )
        self.assertEqual(len(find_indices_equal_to(self.test_match_arr, 9)), 0)
        B = np.copy(self.test_match_arr)
        B[1, :] = 9
        self.assertTrue(
            np.array_equal(
                find_indices_equal_to(B, 9),
                np.array([[1, 0], [1, 1], [1, 2]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
