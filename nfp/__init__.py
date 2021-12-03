from .layers import *
from .models import *
from .preprocessing import *


class MissingDependencyException(RuntimeError):
    pass


custom_objects = {
    'Gather': Gather,
    'Reduce': Reduce,
    'masked_mean_squared_error': masked_mean_squared_error,
    'masked_mean_absolute_error': masked_mean_absolute_error,
    'masked_log_cosh': masked_log_cosh,
    'EdgeUpdate': EdgeUpdate,
    'NodeUpdate': NodeUpdate,
    'GlobalUpdate': GlobalUpdate
}

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
