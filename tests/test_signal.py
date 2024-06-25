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
        with pytest.raises(IndexError):
            idx = np.array([1,2], dtype=np.float64)
            a[idx] = 0
    
    def test_invalid_setitem_value(self, sample_signal):
        a = sample_signal
        with pytest.raises(AssertionError):
            a[1] = '1'
        a[1]=10
        assert a[1]==10

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

    def test_update(self, sample_signal):
        new_data = np.arange(12)
        data_updated = sample_signal.update(data=new_data)
        assert all(new_data==data_updated.data)

        new_fs = 2.0
        fs_updated = sample_signal.update(fs=new_fs)
        assert new_fs==fs_updated.fs

        with pytest.raises(AssertionError):
            wrong_input = sample_signal.update(invalid_arg=3.0)
        
        # Test timestamps are reset
        tstamps = sample_signal.time
        assert sample_signal.tstamps is not None
        data_updated = sample_signal.update(data=new_data, tstamps=np.arange(12))
        assert data_updated.tstamps is None or len(data_updated.tstamps)==0

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

class TestSignalOps:
    # TODO add tests with Signal and np.arrays as other element
    @pytest.fixture
    def original_signal(self):
        test_signal = Signal(data=np.array([1,0,0,1,1,0,0,1]))
        yield test_signal

    @pytest.fixture
    def simple_range_signal(self):
        test_signal = Signal(data=np.array([1,2,3,4,5,6,7,8,9]))
        yield test_signal

    def test_sum(self, original_signal):
        expected_pos = np.array([2,1,1,2,2,1,1,2])
        expected_sub = np.array([0,-1,-1,0,0,-1,-1,0])
        
        add_right = original_signal + 1
        assert np.array_equal(expected_pos, add_right)
        add_left = 1 + original_signal
        assert np.array_equal(expected_pos, add_left)

        add_neg_right = original_signal + (-1)
        assert np.array_equal(expected_sub, add_neg_right)
        add_neg_left = (-1) + original_signal
        assert np.array_equal(expected_sub, add_neg_left)

    def test_sub(self, original_signal):
        expected_right = np.array([0,-1,-1,0,0,-1,-1,0])
        expected_left = np.array([0,1,1,0,0,1,1,0])

        sub_right = original_signal - 1
        assert np.array_equal(expected_right, sub_right)
        sub_left = 1 - original_signal
        assert np.array_equal(expected_left, sub_left)

    def test_mul(self, original_signal):
        expected = np.array([2,0,0,2,2,0,0,2])
        mul_right = original_signal * 2
        assert np.array_equal(expected, mul_right)
        mul_left = 2 * original_signal
        assert np.array_equal(expected, mul_left)

    def test_div(self, original_signal):
        expected_right = np.array([0.5,0,0,0.5,0.5,0,0,0.5])
        div_right = original_signal / 2
        assert np.array_equal(expected_right, div_right) 

        expected_left = np.array([2,np.inf,np.inf,2,2,np.inf,np.inf,2])
        with pytest.warns(RuntimeWarning):
            div_left_byzero = 2 / original_signal
            assert np.array_equal(expected_left, div_left_byzero)

    def test_pow(self, simple_range_signal):
        # Original [1,2,3,4,5,6,7,8,9]
        # **2
        expected = np.array([1,4,9,16,25,36,49,64,81])
        pow_2 = simple_range_signal**2
        assert np.array_equal(expected, pow_2)

        # **0.5 (sqrt)
        expected = np.array([1., 1.41421356, 1.73205081, 2., 2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.])
        pow_05 = simple_range_signal**0.5
        assert np.isclose(expected, pow_05, atol=1e-5).all()

        # 2**sig
        expected = np.array([2,4,8,16,32,64,128,256,512])
        two_pow = 2**simple_range_signal
        assert np.array_equal(expected, two_pow)