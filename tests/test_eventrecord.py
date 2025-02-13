from datetime import datetime, timedelta

import numpy as np
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

    def test_empty_asarray(self, empty_binary_record):
        # An empty EventRecord should be represented by an empty array
        empty_array = empty_binary_record.as_array(sampling_period = 1.0)
        assert np.all(empty_array.data == np.zeros(0, dtype=empty_array.data.dtype))

    def test_different_start_asarray(self, simple_binary_record):
        # Change start date
        simple_binary_record.start_time = simple_binary_record.start_time-timedelta(seconds=1)
        simple_array = simple_binary_record.as_array(sampling_period=1)
        assert np.all(simple_array.data[0:2] == simple_binary_record.start_value)

    def test_memory_collision(self):
        # The data of two EventRecord should have different addresses. Bug caused by dataclass hashable default types. Fixed in v0.2.7
        a = EventRecord(label='a', start_time=datetime.fromtimestamp(0), start_value=0, is_binary=False, is_spikes=False)
        b = EventRecord(label='b', start_time=datetime.fromtimestamp(0), start_value=0, is_binary=False, is_spikes=False)

        assert id(a.data)!=id(b.data)

    def test_xml_valuefunction(self):
        # The data from an XML with a value function should be different from a binary one with just the duration of the event (See test_eventrecord.xml)
        test_xml = './tests/sample_eventrecord.xml'
        t0 = datetime.fromtimestamp(0)
        data_function = EventRecord.from_xml(test_xml, label='d', event_key='EventConcept',target_values='SpO2 desaturation|SpO2 desaturation',\
                                               ts_key='Start',t0=t0,start_value=0, duration_key='Duration', events_path=['PSGAnnotation', 'ScoredEvents', 'ScoredEvent'],\
                                               ts_is_datetime=False, value_function=lambda x:float(x['SpO2Baseline'])-float(x['SpO2Nadir']))
        data_nofunction = EventRecord.from_xml(test_xml, label='d', event_key='EventConcept',target_values='SpO2 desaturation|SpO2 desaturation',\
                                               ts_key='Start',t0=t0,start_value=0, duration_key='Duration', events_path=['PSGAnnotation', 'ScoredEvents', 'ScoredEvent'],\
                                               ts_is_datetime=False)
        assert data_function.is_binary!=data_nofunction.is_binary
        assert all(x<=1 for _,x in data_nofunction)
        assert all(x==0 or np.isclose(x, 2.0) for _,x in data_function)