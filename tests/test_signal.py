from datetime import datetime, date, time

import numpy as np
import pytest

from physio_cassette.physio_cassette import Signal
from scipy.io import savemat
import os

class TestSignal:
    @pytest.fixture
    def sample_signal(self):
        yield Signal(label='test', data=np.arange(10), fs=1.0)

    @pytest.fixture
    def shorter_signal(self):
        yield Signal(label='test', data=np.arange(9), fs=1.0)
    
    @pytest.fixture
    def different_fs_signal(self):
        yield Signal(label='test', data=np.arange(10), fs=2.0)

    def test_invalid_fs(self):
        with pytest.raises(AssertionError):
            Signal(label='test', data=np.arange(10), fs=np.nan)
        with pytest.raises(AssertionError):
            Signal(label='test', data=np.arange(10), fs=-1)
    
    def test_invalid_setitem_index(self, sample_signal):
        a = sample_signal
        with pytest.raises(AssertionError):
            idx = np.array([1,2], dtype=np.float64)
            a[idx] = 0
    
    def test_invalid_setitem_value(self, sample_signal):
        a = sample_signal
        with pytest.raises(AssertionError):
            a[1] = '1'

    def test_invalid_sub(self, sample_signal):
        a = sample_signal
        with pytest.raises(AssertionError):
            a.sub(start=1, indexes=slice(1,2, None))
        with pytest.raises(AssertionError):
            a.sub(1)
        with pytest.raises(AssertionError):
            a.sub(indexes=1)
        with pytest.raises(AssertionError):
            a.sub(indexes=[1,2,3,4])

    def test_invalid_ops(self, sample_signal, shorter_signal, different_fs_signal):
        a = sample_signal
        b = shorter_signal
        c = different_fs_signal
        with pytest.raises(ValueError):
            a+np.array([1,2,3])
        with pytest.raises(ValueError):
            a+c
        with pytest.warns(UserWarning):
            a+b

class TestSignalFromMat:
    @pytest.fixture
    def good_file(self):
        filename = "./test.mat"
        savemat(filename, {
            "data": np.ones((10,1)),
            "SampleRate": 1,
            "StartDate": "1/1/2000",
            "StartTime": "12:12:12"
        })
        yield filename
        os.remove(filename)

    @pytest.fixture
    def missing_file(self):
        filename = "./test.mat"
        savemat(filename, {
            "data": np.ones((10,1)),
            "SampleRate": 1,
            "StartDate": "1/1/2000"
        })
        yield filename
        os.remove(filename)

    def test_wrong_format(self):
        with pytest.raises(AssertionError):
            a,_ = Signal.from_mat_file("./test.csv")
    
    def test_missing_data(self, missing_file):
        with pytest.warns(UserWarning):
            a,_ = Signal.from_mat_file(missing_file)
        assert a is None