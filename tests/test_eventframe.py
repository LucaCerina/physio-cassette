from datetime import datetime, timedelta

import pytest

from physio_cassette.physio_cassette import EventRecord, EventFrame
from traces import TimeSeries

class TestEventFrame:
    @pytest.fixture
    def empty_binary_frame(self):
        labels = ['a', 'b', 'c']
        t0 = datetime.fromtimestamp(0)
        output = EventFrame(start_date=t0)
        with pytest.warns(UserWarning):
            for label in labels:
                output[label] = EventRecord.from_state_array('empty_binary', t0=t0, input_array=[], start_value=0)
        yield output

    @pytest.fixture
    def small_binary_frame(self):
        labels = ['a', 'b', 'c']
        t0 = datetime.fromtimestamp(0)
        output = EventFrame(start_date=t0)
        for label in labels:
            output[label] = EventRecord.from_state_array('empty_binary', t0=t0, input_array=[1,0,1,0,1,0], start_value=0)
        yield output

    def test_n_events(self, small_binary_frame, empty_binary_frame):
        assert empty_binary_frame.n_events == 0
        assert small_binary_frame.n_events == 9

    def test_merged_labels(self, small_binary_frame):
        # Test missing label
        with pytest.warns(UserWarning):
            small_binary_frame.merged_data(labels=['d'])