"""Example test module."""

import aind_anatomical_utils


def test_version():
    """Test that version is defined."""
    assert aind_anatomical_utils.__version__ is not None
    assert isinstance(aind_anatomical_utils.__version__, str)
