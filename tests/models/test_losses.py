import pytest
import numpy as np

from keras import backend as K
from nfp.models import (masked_mean_absolute_error,
                        masked_mean_squared_error,
                        masked_log_cosh)

def test_losses():

    a = np.random.random((5, 6, 7))
    b = np.random.random((5, 6, 7))

    a[2, 5, 2] = np.nan
    a[3, 1, 0] = np.nan

    y_a = K.variable(a)
    y_b = K.variable(b)

    assert np.isfinite(K.eval(masked_mean_absolute_error(y_a, y_b)))
    assert np.isfinite(K.eval(masked_mean_squared_error(y_a, y_b)))
    assert np.isfinite(K.eval(masked_log_cosh(y_a, y_b)))
