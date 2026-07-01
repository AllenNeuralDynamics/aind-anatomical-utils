"""Tools to work with anatomically-aware files."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aind-anatomical-utils")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
