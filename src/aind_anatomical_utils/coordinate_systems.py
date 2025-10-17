"""Module to deal with coordinate systems"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Final, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type aliases - using simple assignment which works with PEP 585


class CoordSys(str, Enum):
    RAS = "RAS"
    LAS = "LAS"
    RPS = "RPS"
    LPS = "LPS"
    RAI = "RAI"
    LAI = "LAI"
    RPI = "RPI"
    LPI = "LPI"


CS = Union[str, CoordSys]

# axis mapping: (axis_index, sign)
# axis_index here is arbitrary but must be consistent between opposite
# directions. Same with sign.
_AXES: Final = {
    "L": (0, +1),
    "R": (0, -1),
    "P": (1, +1),
    "A": (1, -1),
    "I": (2, -1),
    "S": (2, +1),
}

_OPPOSITE_AXES: Final = {
    "L": "R",
    "R": "L",
    "P": "A",
    "A": "P",
    "I": "S",
    "S": "I",
}


def _norm_code(code: CS) -> str:
    return code.value if isinstance(code, CoordSys) else str(code).upper()


def _validate_code(code: str) -> None:
    """Validates a coordinate system string.

    Ensures that each character in the coordinate system string belongs to the
    set 'R/L', 'A/P', or 'I/S' and that no axis or its opposite is repeated.

    Parameters
    ----------
    coord : str
        The coordinate system string to validate.

    Returns
    -------
    Set[str]
        A set of unique directions in the coordinate system string.
    """
    seen_axes = set()
    for c in code:
        if c not in _AXES:
            raise ValueError(f"Direction '{c}' not in R/L, A/P, or I/S")
        ax, _ = _AXES[c]
        if ax in seen_axes:
            raise ValueError(f"Axis for '{c}' not unique in code '{code}'")
        seen_axes.add(ax)


@dataclass(frozen=True)
class Orientation:
    __slots__ = ("perm", "sign", "R", "det")
    perm: NDArray[np.intp]  # (N,) permutation indices
    sign: NDArray[np.int8]  # (N,) ±1
    R: NDArray[np.float64]  # (N,N) orthonormal reorientation matrix
    det: float  # determinant of R (±1)


@cache
def _orientation(src: str, dst: str) -> Orientation:
    if len(src) != len(dst):
        raise ValueError("Source and destination must have same length")
    _validate_code(src)
    _validate_code(dst)

    # Build R by rows: for each dst letter, pick the src axis and sign
    N = len(src)
    R = np.zeros((N, N), dtype=np.float64)
    perm = np.empty(N, dtype=np.intp)
    sign = np.empty(N, dtype=np.int8)

    # map from src letters to (axis_index, sign)
    # e.g., 'R' -> (0,+1), 'L' -> (0,-1), etc.
    # src_axis = {c: _AXES[c][0] for c in src}
    for i, d in enumerate(dst):
        d_ax, d_sgn = _AXES[d]
        # find letter in src whose axis matches d_ax (either direction)
        # we scan src to find the axis match and the sign from src letter
        for j, s in enumerate(src):
            s_ax, s_sgn = _AXES[s]
            if s_ax == d_ax:
                perm[i] = j
                sign[i] = 1 if d_sgn == s_sgn else -1
                R[i, j] = float(sign[i])
                break
        else:
            raise ValueError(
                f"Destination direction '{d}' has no match in source '{src}'"
            )
    det = float(round(np.linalg.det(R)))  # should be ±1
    return Orientation(perm=perm, sign=sign, R=R, det=det)


def coordinate_transform_matrix(src: CS, dst: CS) -> NDArray[np.float64]:
    """Return orthonormal matrix R mapping points in src->dst

    (row-major: p_dst = p_src @ R.T).
    """
    src_n, dst_n = _norm_code(src), _norm_code(dst)
    return _orientation(src_n, dst_n).R


def find_coordinate_perm_and_flips(
    src: CS, dst: CS
) -> tuple[NDArray[np.intp], NDArray[np.int8]]:
    """Determine how to convert between coordinate systems.

    This function takes a source `src` and destination `dst` string specifying
    two coordinate systems, and finds the permutation and sign flip such that a
    source array can be transformed from its own coordinate system to the
    destination coordinate system by first applying the permutation and then
    multiplying the resulting array by the direction. That is, the input can
    be transformed to the desired coordinate system with the following code:
    `dst_array = direction * src_array[:, perm]`, where `direction` and `perm`
    are the returned values of this function.

    Coordinate system are defined by strings specifying how each axis aligns to
    anatomical directions, with each character belonging to the set 'APLRIS',
    corresponding to Anterior, Posterior, Left, Right, Inferior, Superior.

    An example string would be 'RAS' corresponding to Right, Anterior, Superior
    for the first, second, and third axes respectively. The axis increases in
    the direction indicated (i.e. 'R' means values are more positive as you
    move to the patient's right).

    Parameters
    ----------
    src : str
        String specifying the source coordinate system, with each character
        belonging to the set 'R/L', 'A/P', or 'I/S'.
    dst : str
        String specifying the destination coordinate system, with each
        character belonging to the set 'R/L', 'A/P', or 'I/S'.

    Returns
    -------
    perm : np.ndarray(dtype=int16) (N)
        Permutation array used to convert the `src` coordinate system to the
        `dst` coordinate system
    direction: np.ndarray(dtype=int16) (N)
        Direction array used to multiply the `src` coordinate system after
        permutation into the `dst` coordinate system

    Raises
    ------
    ValueError
        If the source or destination coordinate systems are invalid or
        incompatible.
    """
    src_n, dst_n = _norm_code(src), _norm_code(dst)
    o = _orientation(src_n, dst_n)
    return o.perm.copy(), o.sign.copy()


def convert_coordinate_system(
    arr: NDArray,
    src_coord: CS,
    dst_coord: CS,
    *,
    axis: int = -1,
    copy: bool = True,
    prefer_matrix: bool | None = None,
) -> NDArray:
    """Converts points in one anatomical coordinate system to another.

    This will permute and multiply the NxM input array `arr` so that N
    M-dimensional points in the coordinate system specified by `src_coord` will
    be transformed into the destination coordinate system specified by
    `dst_coord`. The current implementation does not allow the dimensions to
    change.

    Coordinate systems are defined by strings specifying how each axis aligns
    to anatomical directions, with each character belonging to the set
    'APLRIS', corresponding to Anterior, Posterior, Left, Right, Inferior,
    Superior, respectively.

    An example string would be 'RAS' corresponding to Right, Anterior, Superior
    for the first, second, and third axes respectively. The axis increases in
    the direction indicated (i.e. 'R' means values are more positive as you
    move to the patient's right).

    Parameters
    ----------
    arr : np.ndarray (N x M)
        N points of M dimensions (at most three).
    src_coord : str
        String specifying the source coordinate system, with each character
        belonging to the set 'R/L', 'A/P', or 'I/S'.
    dst_coord : str
        String specifying the destination coordinate system, with each
        character belonging to the set 'R/L', 'A/P', or 'I/S'.
    axis : int, optional
        The axis of `arr` that corresponds to the point dimensions, by default
        -1
    copy : bool, optional
        Whether to return a copy of the data even if no transformation is
        necessary, by default True.
    prefer_matrix : bool, optional
        Whether to prefer using matrix multiplication for the transformation.
        If None (default), will use matrix multiplication if the array is of
        floating type and has more than one point.

    Returns
    -------
    np.ndarray (N x M)
        The N input points transformed into the destination coordinate system.

    Raises
    ------
    ValueError
        If the source or destination coordinate systems are invalid or
        incompatible.
    """
    src_n, dst_n = _norm_code(src_coord), _norm_code(dst_coord)
    if src_n == dst_n:
        return arr.copy() if copy else arr

    o = _orientation(src_n, dst_n)

    if prefer_matrix is None:
        prefer_matrix = np.issubdtype(arr.dtype, np.floating) and arr.ndim > 1

    if prefer_matrix:
        arr_m = np.moveaxis(arr, axis, -1)
        out = arr_m @ o.R.T
        return np.moveaxis(out, -1, axis)

    out = np.take(arr, o.perm, axis=axis)  # copy
    # broadcast sign along `axis`
    shape = [1] * out.ndim
    shape[axis] = len(o.sign)
    out *= o.sign.reshape(shape)
    return out


def reorient_mesh_vertices_faces(
    vertices: NDArray[np.floating],
    faces: NDArray[np.integer],
    src: CS,
    dst: CS,
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Reorient mesh vertices from src->dst and flip face winding if
    reflection."""
    src_n, dst_n = _norm_code(src), _norm_code(dst)
    orientation = _orientation(src_n, dst_n)
    R = orientation.R
    v2 = vertices @ R.T
    det = orientation.det
    if det < 0:
        f2 = faces[:, ::-1].copy()
    else:
        f2 = faces.copy()
    return v2, f2
