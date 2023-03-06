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