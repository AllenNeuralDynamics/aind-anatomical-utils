"""Guard the package's ANTs-free import contract.

``antspyx`` is an optional extra (``aind-anatomical-utils[ants]``). The ANTs-free
modules and ``anatomical_volume`` must import without it; only the ANTs code
paths (``ants_volume`` and ``AnatomicalHeader.as_ants``) may require it.
"""

import subprocess
import sys


def test_anatomical_volume_import_does_not_pull_ants() -> None:
    """Importing anatomical_volume must not import the ``ants`` module.

    A top-level ``import ants`` re-introduced into ``anatomical_volume`` would
    silently make ANTs a hard dependency again. Runs in a fresh subprocess so an
    ``ants`` already imported by the parent test session cannot mask a regression.
    """
    code = (
        "import sys\n"
        "import aind_anatomical_utils.anatomical_volume\n"
        "assert 'ants' not in sys.modules, 'anatomical_volume imported ants at module load'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ants_free_public_modules_import() -> None:
    """The ANTs-free public modules must import without antspyx present."""
    code = (
        "import aind_anatomical_utils\n"
        "from aind_anatomical_utils import slicer, coordinate_systems, sitk_volume, utils\n"
        "from aind_anatomical_utils.anatomical_volume import AnatomicalHeader, fix_corner_compute_origin\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
