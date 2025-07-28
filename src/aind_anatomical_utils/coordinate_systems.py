"""Module to deal with coordinate systems"""

from typing import Dict, Set, Tuple

import numpy as np
from numpy.typing import NDArray


def _validate_coordinate_system(
    coord: str, pairs: Dict[str, str], coord_type: str
) -> Set[str]:
    """Validates a coordinate system string.

    Ensures that each character in the coordinate system string belongs to the
    set 'R/L', 'A/P', or 'I/S' and that no axis or its opposite is repeated.

    Parameters
    ----------
    coord : str
        The coordinate system string to validate.
    pairs : Dict[str, str]
        A dictionary mapping each direction to its opposite.
    coord_type : str
        A label for the coordinate system being validated (e.g., "Source" or
        "Destination").

    Returns
    -------
    Set[str]
        A set of unique directions in the coordinate system string.
    """
    coord_set = set()
    for i, c in enumerate(coord):
        if c not in pairs:
            raise ValueError(
                f"{coord_type} direction '{c}' not in R/L, A/P, or I/S"
            )
        if c in coord_set or pairs[c] in coord_set:
            raise ValueError(f"{coord_type} axis '{c}' not unique")
        coord_set.add(c)
    return coord_set


def _build_src_order(src: str, pairs: Dict[str, str]) -> Dict[str, int]:
    """Builds a mapping of source directions to their indices.

    Parameters
    ----------
    src : str
        The source coordinate system string.
    pairs : Dict[str, str]
        A dictionary mapping each direction to its opposite.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping each direction in the source coordinate system
        to its index.
    """
    src_order = {}
    for i, s in enumerate(src):
        if s not in pairs:
            raise ValueError(f"Source direction '{s}' not in R/L, A/P, or I/S")
        if s in src_order or pairs[s] in src_order:
            raise ValueError(f"Source axis '{s}' not unique")
        src_order[s] = i
    return src_order


def find_coordinate_perm_and_flips(
    src: str, dst: str
) -> Tuple[NDArray[np.int16], NDArray[np.int16]]:
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
    nel = len(src)
    if len(dst) != nel:
        raise ValueError("Inputs should be the same length")
    src_u, dst_u = src.upper(), dst.upper()
    basic_pairs = dict(R="L", A="P", S="I")
    pairs = {**basic_pairs, **{v: k for k, v in basic_pairs.items()}}

    src_order = _build_src_order(src_u, pairs)
    _validate_coordinate_system(dst_u, pairs, "Destination")

    perm = -1 * np.ones(nel, dtype="int16")
    direction = np.zeros(nel, dtype="int16")
    for i, d in enumerate(dst_u):
        if d in src_order:
            perm[i] = src_order[d]
            direction[i] = 1
        elif pairs[d] in src_order:
            perm[i] = src_order[pairs[d]]
            direction[i] = -1
        else:
            raise ValueError(
                f"Destination direction '{d}' has no match in source "
                f"directions '{src_u}'"
            )
    return perm, direction


def convert_coordinate_system(
    arr: NDArray, src_coord: str, dst_coord: str
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
    perm, direction = find_coordinate_perm_and_flips(src_coord, dst_coord)
    if arr.ndim == 1:
        out = arr[perm]
    else:
        out = arr[:, perm]
    out *= direction
    return out
