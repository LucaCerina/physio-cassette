from datetime import datetime, timedelta

import pytest

from physio_cassette.physio_cassette import EventRecord
from traces import TimeSeries


class TestEventRecord:
    @pytest.fixture
    def len_test_records(self):
        yield 10

    @pytest.fixture
    def empty_record_nostart(self):
        t0 = datetime.fromtimestamp(0)
        input_array = []
        with pytest.warns(UserWarning):
            yield EventRecord.from_state_array('empty_binary_nostart', t0=t0, input_array=input_array)
    
    @pytest.fixture
    def empty_binary_record(self):
        t0 = datetime.fromtimestamp(0)
        input_array = []
        with pytest.warns(UserWarning):
            yield EventRecord.from_state_array('empty_binary', t0=t0, input_array=input_array, start_value=0)

    @pytest.fixture
    def simple_record(self, len_test_records):
        t0 = datetime.fromtimestamp(0)
        input_array = [x for x in range(len_test_records)]
        yield EventRecord.from_state_array('empty_binary', t0=t0, input_array=input_array, start_value=0)

    @pytest.fixture
    def simple_binary_record(self, len_test_records):
        # Simple binary array
        t0 = datetime.fromtimestamp(0)
        input_array = [1,0]*len_test_records
        yield EventRecord.from_state_array('simple_binary', t0=t0, input_array=input_array, start_value=0)
    
    @pytest.fixture
    def simple_binary_nostart_record(self, len_test_records):
        # Binary array with start value covered by data
        t0 = datetime.fromtimestamp(0)
        input_array = [1,0]*len_test_records
        ts = [t0+timedelta(seconds=i) for i in range(len(input_array))]
        yield EventRecord.from_state_array('simple_binary_nostart', t0=t0, input_array=input_array, ts_array=ts, start_value=0)   

    def test_length(self, empty_binary_record, simple_record, simple_binary_record, simple_binary_nostart_record, len_test_records):
        assert empty_binary_record.n_events==0
        assert simple_record.n_events==len_test_records
        # Binary records return correct number of events even if start value changes
        assert simple_binary_record.n_events==len_test_records
        assert simple_binary_nostart_record.n_events==len_test_records
    
    def test_duration(self, empty_record_nostart, empty_binary_record, simple_record):
        assert empty_record_nostart.duration == 0
        assert empty_binary_record.duration == 0
        assert simple_record.duration == 9

    def test_post_init(self, empty_binary_record):
        # Empty record with only start value will have the start_value in it
        assert empty_binary_record.data.items()[0][1] == 0

        # Create eventrecord with 1 value and start value
        # from_state_array skips post_init checks by setting start_value directly at t0
        test_record = EventRecord.from_state_array('test', t0=datetime.fromtimestamp(0), input_array=[1], ts_array=[1], start_value=0)
        assert test_record.data.items()[0][1] == 0
        test_record = EventRecord.from_state_array('test', t0=datetime.fromtimestamp(0), input_array=[1], ts_array=[1], start_value=None)
        assert test_record.data.items()[0][1] != 0
        # Force post_init call
        test_record = EventRecord(start_time=datetime.fromtimestamp(0), data=TimeSeries(data={datetime.fromtimestamp(1):1}), start_value=0)
        assert test_record.data.items()[0][1] == 0