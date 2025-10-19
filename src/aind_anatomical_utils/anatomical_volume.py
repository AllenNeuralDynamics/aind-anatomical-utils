"""
Functions for working with the headers of anatomical volumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, cast

import ants  # type: ignore[import-untyped]
import numpy as np
import SimpleITK as sitk
from numpy.typing import DTypeLike, NDArray

from aind_anatomical_utils.coordinate_systems import (
    _OPPOSITE_AXES,
    _norm_code,
    convert_coordinate_system,
    find_coordinate_perm_and_flips,
)

if TYPE_CHECKING:
    from ants.core import ANTsImage  # type: ignore[import-untyped]

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class AnatomicalHeader:
    """
    Lightweight, immutable wrapper of an ITK/SimpleITK header.

    The physical mapping is:

    ``physical = origin + direction @ (spacing ⊙ index)``

    where **columns** of ``direction`` are the LPS unit vectors of the index
    axes ``(i, j, k)``; ``spacing`` is per-index-axis (mm); and ``origin`` is
    LPS (mm).

    Parameters
    ----------
    origin : tuple of float
        Image origin in LPS (mm). Length-3.
    spacing : tuple of float
        Spacing per **index axis** ``(i, j, k)`` in millimeters. Length-3.
    direction : numpy.ndarray
        3×3 direction cosine matrix. Columns are unit vectors for ``i, j, k``
        expressed in LPS. Row-major when flattened for SimpleITK APIs.
    size_ijk : tuple of int
        Image size (number of voxels) along ``(i, j, k)``. Used by overlays
        like corner anchoring.

    Notes
    -----
    - Use :meth:`as_sitk` to obtain a 1×1×1 SimpleITK image carrying this
      header.
    - Use :meth:`update_sitk` to set the header onto an existing image.
    """

    origin: Vec3  # LPS mm
    spacing: Vec3  # per INDEX axis (i,j,k) in mm
    # (3,3) columns are unit vectors for i,j,k (in LPS)
    direction: NDArray[np.floating]
    size_ijk: tuple[int, int, int]  # needed for corner anchoring etc.

    def direction_tuple(self) -> tuple[float, ...]:
        """
        Return the direction matrix flattened row-major as a tuple of floats.

        Returns
        -------
        tuple of float
            Length-9 row-major flattening of the 3×3 direction matrix, suitable
            for :meth:`SimpleITK.Image.SetDirection`.
        """
        return tuple(float(x) for x in self.direction.ravel())

    def orientation_code(self) -> str:
        """
        Get the anatomical orientation code (e.g., 'RAS', 'LPI') of this
        header.

        Returns
        -------
        str
            The 3-letter anatomical orientation code corresponding to the
            direction matrix of this header.
        """
        direction_code = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                self.direction_tuple()
            )
        )
        direction_code = cast(str, direction_code)
        return direction_code

    def is_axis_aligned(self, tol: float = 1e-6) -> bool:
        """
        Check if the anatomical volume is axis-aligned.

        Parameters
        ----------
        tol : float
            Tolerance for checking alignment.

        Returns
        -------
        bool
            True if the volume is axis-aligned, False otherwise.
        """
        A = self.direction
        r_idx = np.argmax(np.abs(A), axis=0)  # shape (3,)
        # Those three rows must be all different (perm)
        if len(set(r_idx.tolist())) != 3:
            return False
        # Build the ideal ±1 one-hot matrix B from the peaks
        B = np.zeros_like(A)
        signs = np.sign(A[r_idx, np.arange(3)])
        B[r_idx, np.arange(3)] = signs
        # Close to ideal one-hot ±1?
        return np.allclose(A, B, atol=tol)

    def update_sitk(self, sitk_image: sitk.Image) -> sitk.Image:
        """
        Set this header (origin, spacing, direction) on a SimpleITK image.

        Parameters
        ----------
        sitk_image : SimpleITK.Image
            The image whose header should be updated.

        Returns
        -------
        SimpleITK.Image
            The same image instance, updated in-place for convenience.
        """
        sitk_image.SetOrigin(tuple(self.origin))
        sitk_image.SetSpacing(tuple(self.spacing))
        sitk_image.SetDirection(self.direction_tuple())
        return sitk_image

    def update_ants(self, ants_image: ANTsImage) -> ANTsImage:
        """
        Set this header (origin, spacing, direction) on an ANTs image.

        Parameters
        ----------
        ants_image : ants.core.ANTsImage
            The image whose header should be updated.

        Returns
        -------
        ants.core.ANTsImage
            The same image instance, updated in-place for convenience.
        """
        ants_image.set_origin(self.origin)
        ants_image.set_spacing(self.spacing)
        ants_image.set_direction(self.direction)
        return ants_image

    def as_sitk_stub(self) -> sitk.Image:
        """
        Create a minimal SimpleITK image (1×1×1) carrying this header.

        Returns
        -------
        SimpleITK.Image
            A new image with this :class:`Header`'s origin, spacing, and
            direction set. Pixel type is ``UInt8`` and size is 1×1×1, which is
            sufficient for coordinate transforms via
            :meth:`TransformContinuousIndexToPhysicalPoint`.
        """
        img = sitk.Image([1, 1, 1], sitk.sitkUInt8)
        self.update_sitk(img)
        return img

    def as_sitk(self, p_type: int = sitk.sitkUInt8) -> sitk.Image:
        """
        Create a SimpleITK image carrying this header.

        Returns
        -------
        SimpleITK.Image
            A new image with this :class:`Header`'s origin, spacing, direction,
            and size set. Pixel type is ``UInt8``.
        """
        img = sitk.Image(list(self.size_ijk), p_type)
        self.update_sitk(img)
        return img

    def as_ants(self, np_eltype: DTypeLike = np.uint8) -> ANTsImage:
        """
        Create an ANTs image carrying this header.

        Returns
        -------
        ants.core.ANTsImage
            A new image with this :class:`Header`'s origin, spacing, direction,
            and size set. Pixel type is ``unsigned char``.
        """
        arr = np.zeros(self.size_ijk, dtype=np_eltype)
        img = ants.from_numpy(
            arr,
            origin=self.origin,
            spacing=self.spacing,
            direction=self.direction,
        )
        return img

    @classmethod
    def from_sitk(
        cls,
        sitk_image: sitk.Image,
        size_ijk: tuple[int, int, int] | None = None,
    ) -> AnatomicalHeader:
        """
        Construct a :class:`Header` from a SimpleITK image.

        Parameters
        ----------
        sitk_image : SimpleITK.Image
            Source image.
        size_ijk : tuple of int or None
            Size to record as ``(i, j, k)``. If ``None``, uses
            :meth:`SimpleITK.Image.GetSize`.

        Returns
        -------
        Header
            New header with origin, spacing, direction, and ``size_ijk`` taken
            from ``sitk_image`` (or the provided size).
        """
        origin = sitk_image.GetOrigin()
        spacing = sitk_image.GetSpacing()
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
        if size_ijk is None:
            size_ijk = cast(tuple[int, int, int], sitk_image.GetSize())

        return cls(
            origin=origin,
            spacing=spacing,
            direction=direction,
            size_ijk=size_ijk,
        )

    @classmethod
    def from_ants(
        cls,
        ants_image: ANTsImage,
        size_ijk: tuple[int, int, int] | None = None,
    ) -> AnatomicalHeader:
        """
        Construct a :class:`Header` from an ANTs image.

        Parameters
        ----------
        ants_image : ants.core.ANTsImage
            Source image.
        size_ijk : tuple of int or None
            Size to record as ``(i, j, k)``. If ``None``, uses
            :meth:`ants.core.ANTsImage.shape`.

        Returns
        -------
        Header
            New header with origin, spacing, direction, and ``size_ijk`` taken
            from ``ants_image`` (or the provided size).
        """
        origin = ants_image.origin
        spacing = ants_image.spacing
        direction = ants_image.direction
        if size_ijk is None:
            size_ijk = ants_image.shape

        return cls(
            origin=origin,
            spacing=spacing,
            direction=direction,
            size_ijk=size_ijk,
        )

    def regrid_to(self, dst_coord_system: str) -> AnatomicalHeader:
        """
        Regrid this header to a destination coordinate system.

        Creates a new header with the same physical voxel locations but
        reoriented to the specified anatomical coordinate system (e.g., "RAS",
        "LPS"). All voxel centers remain at identical physical locations.

        Parameters
        ----------
        dst_coord_system : str
            The destination coordinate system (e.g., "RAS", "LPS").

        Returns
        -------
        AnatomicalHeader
            A new header with the same physical space but regridded to the
            destination coordinate system.

        Raises
        ------
        ValueError
            If this header is not axis-aligned.

        Examples
        --------
        >>> ras_header = AnatomicalHeader(...)  # RAS orientation
        >>> lps_header = ras_header.regrid_to("LPS")
        >>> # All voxel centers remain at same physical locations

        Notes
        -----
        This operation preserves the physical bounding box and ensures that
        every voxel center in the new grid corresponds exactly to a voxel
        center in the original grid.
        """
        if not self.is_axis_aligned():
            raise ValueError("Source header must be axis-aligned.")

        src_code = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                tuple(self.direction.flatten())
            )
        )
        perm, _ = find_coordinate_perm_and_flips(src_code, dst_coord_system)
        dir_tup_new = (
            sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
                dst_coord_system
            )
        )
        D_new = np.asarray(dir_tup_new).reshape(3, 3)
        size_new = tuple(self.size_ijk[i] for i in perm)
        size_new = cast(tuple[int, int, int], size_new)
        spacing_new = tuple(self.spacing[i] for i in perm)
        spacing_new = cast(tuple[float, float, float], spacing_new)

        # The corner code of the original origin is the opposite of the
        # src_code
        old_origin_corner_code = "".join(
            _OPPOSITE_AXES[src_code[i]] for i in range(3)
        )
        origin_new, _, _ = fix_corner_compute_origin(
            size=size_new,
            spacing=spacing_new,
            direction=D_new,
            target_point=self.origin,
            corner_code=old_origin_corner_code,
            use_outer_box=False,
        )
        return AnatomicalHeader(
            size_ijk=size_new,
            spacing=spacing_new,
            origin=origin_new,
            direction=D_new,
        )


def _corner_indices(
    size: NDArray[np.integer], outer: bool = True
) -> NDArray[np.floating]:
    size = np.asarray(size, float)
    lo = -0.5 if outer else 0.0
    hi = (size - 0.5) if outer else (size - 1.0)
    return np.array(
        list(product([lo, hi[0]], [lo, hi[1]], [lo, hi[2]])), float
    )


def fix_corner_compute_origin(
    size: Sequence[int],
    spacing: Sequence[float],
    direction: NDArray[np.floating],
    target_point: Sequence[float],
    corner_code: str = "RAS",
    target_frame: str = "LPS",
    use_outer_box: bool = False,
) -> tuple[tuple[float, float, float], NDArray[np.floating], int]:
    """
    Compute the image origin such that a specified corner of the image
    aligns with a given physical point in a specified coordinate frame.

    Parameters
    ----------
    size : Sequence[int]
        The image size along each spatial axis (e.g., [nx, ny, nz]).
    spacing : Sequence[float]
        The voxel spacing along each axis in millimeters (e.g., [sx, sy, sz]).
    direction : NDArray[np.floating]
        3x3 direction cosine matrix (row-major) in ITK/LPS convention.
    target_point : Sequence[float]
        Physical coordinates (in mm) of the desired corner in the target frame.
    corner_code : str, optional
        3-letter code specifying which image corner to align (e.g., "LPI",
        "RAS").  Default is "LPI".
    target_frame : str, optional
        3-letter code specifying the coordinate frame of `target_point`.
        Defaults to `LPS`.
    use_outer_box : bool, optional
        If True, use bounding box corners (-0.5, size-0.5); if False, use voxel
        centers (0, size-1).  Default is False.

    Returns
    -------
    origin_lps : tuple of float
        The computed image origin in LPS coordinates (mm).
    chosen_corner_index : NDArray[np.floating]
        The continuous index (ijk) of the chosen corner.
    corner_idx_number : int
        The index (0..7) of the chosen corner.

    Notes
    -----
    This function is useful for setting the image origin so that a particular
    image corner matches a desired physical location, taking into account
    direction cosines and coordinate conventions.
    """
    # Normalize to 3D
    size_arr = np.array(list(size) + [1, 1, 1])[:3].astype(float)
    spacing_arr = np.array(list(spacing) + [1, 1, 1])[:3].astype(float)
    target_point_arr = np.array(list(target_point) + [1, 1, 1])[:3].astype(
        float
    )
    D = np.asarray(direction, float).reshape(3, 3)

    # All 8 corners in continuous index space and their LPS offsets from origin
    corners_idx = _corner_indices(size_arr, outer=use_outer_box)  # (8,3)
    offsets_lps = (corners_idx * spacing_arr) @ D.T  # (8,3)

    _, coord_sign = find_coordinate_perm_and_flips(corner_code, "LPS")
    # Pick the corner that is "most" along the requested code axes
    vals = offsets_lps * coord_sign  # convert to that code's axis sense
    # lexicographic argmax: prioritize x, then y, then z in that code
    idx = np.lexsort((vals[:, 2], vals[:, 1], vals[:, 0]))[-1]
    corner_offset_lps = offsets_lps[idx]

    # Convert target point to LPS and solve: target = origin + corner_offset
    target_frame_n = _norm_code(target_frame)
    if target_frame_n == "LPS":
        target_lps = target_point_arr
    else:
        target_lps = convert_coordinate_system(
            target_point_arr, target_frame, "LPS"
        )
    origin_lps = target_lps - corner_offset_lps

    return tuple(origin_lps), corners_idx[idx], idx
