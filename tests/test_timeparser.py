from datetime import datetime

import pytest

from physio_cassette.physio_cassette import parse_timestamp


class TestTimestampParser:
    def test_isoformat(self):
        start_date = datetime(2000,1,1)
        true_date = datetime(2000,1,1,15,0,0)
        test_date = "2000-01-01T15:00:00"
        parsed_date = parse_timestamp(test_date, start_date)
        assert true_date == parsed_date

    def test_pastdate_isoformat(self):
        start_date = datetime(2000,1,1)
        true_date = datetime(2000,1,1,23,0,0)
        test_date = "1999-12-31T23:00:00"
        parsed_date = parse_timestamp(test_date, start_date)
        assert true_date == parsed_date

    def test_hhmmss(self):
        start_date = datetime(2000,1,1)
        true_date = datetime(2000,1,1,15,0,0)
        test_date = "15:00:00"
        parsed_date = parse_timestamp(test_date, start_date)
        assert true_date == parsed_date
        test_date = "15.00.00"
        parsed_date = parse_timestamp(test_date, start_date)
        assert true_date == parsed_date

    def test_fuzzy(self):
        start_date = datetime(2000,1,1)
        true_date = datetime(2000,1,1,8,21,0)
        test_date = "January 1, 2000 at 8:21:00AM"
        parsed_date = parse_timestamp(test_date, start_date)
        assert true_date == parsed_date

    def test_currently_unsupported(self):
        start_date = datetime(2000,1,1)
        true_date = datetime(2000,1,1,8,21,0)
        test_date = "some time in the morning"
        with pytest.raises(ValueError):
            parsed_date = parse_timestamp(test_date, start_date)

    def test_short_string(self):
        start_date = datetime(2000,1,1)
        parsed_date = parse_timestamp("", start_date)
        assert parsed_date is None

    def test_datetime_as_input(self):
        start_date = datetime(2000,1,1)
        with pytest.warns(UserWarning):
            parsed_date = parse_timestamp(start_date, start_date)