
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import (adjacency, graphs, normal, spectrum, utilities)