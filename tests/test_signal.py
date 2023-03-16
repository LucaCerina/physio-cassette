from datetime import datetime

import numpy as np
import pytest

from physio_cassette.physio_cassette import Signal


class TestSignal:
    def test_invalid_fs(self):
        with pytest.raises(AssertionError):
            Signal(label='test', data=np.arange(10), fs=np.nan)
        with pytest.raises(AssertionError):
            Signal(label='test', data=np.arange(10), fs=-1)
    
    def test_invalid_setitem_index(self):
        a = Signal(label='test', data=np.arange(10), fs=1.0)
        with pytest.raises(AssertionError):
            idx = np.array([1,2], dtype=np.float64)
            a[idx] = 0