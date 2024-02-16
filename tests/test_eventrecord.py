from datetime import datetime, timedelta

import pytest

from physio_cassette.physio_cassette import EventRecord


class TestEventRecord:
    @pytest.fixture
    def len_binary(self):
        yield 10

    @pytest.fixture
    def simple_binary_record(self, len_binary):
        # Simple binary array
        t0 = datetime.fromtimestamp(0)
        input_array = [1,0]*len_binary
        yield EventRecord.from_state_array('simple_binary', t0=t0, input_array=input_array, start_value=0)
    
    @pytest.fixture
    def simple_binary_nostart_record(self, len_binary):
        # Binary array with start value covered by data
        t0 = datetime.fromtimestamp(0)
        input_array = [1,0]*len_binary
        ts = [t0+timedelta(seconds=i) for i in range(len(input_array))]
        yield EventRecord.from_state_array('simple_binary', t0=t0, input_array=input_array, ts_array=ts, start_value=0)   

    def test_length(self, simple_binary_record, simple_binary_nostart_record, len_binary):

        # Binary records return correct number of events even if start value changes
        assert simple_binary_record.n_events==len_binary
        assert simple_binary_nostart_record.n_events==len_binary