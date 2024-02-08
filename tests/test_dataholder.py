import pytest
from physio_cassette import DataHolder, Signal

class TestDataHolder:
    @pytest.fixture
    def sample_data(self):
        sig = Signal(label='signal_test', data=[1,2,3])
        yield {'a':1, 'b':2, 'c':3, 'signal':sig}

    @pytest.fixture
    def sample_holder(self, sample_data):
        yield DataHolder(sample_data)

    def test_init(self):
        holder = DataHolder()
        assert issubclass(type(holder), dict)

    def test_iter(self, sample_data, sample_holder):
        for item_holder, item_data in zip(sample_holder, sample_data.items()):
            assert item_holder == item_data

    def test_relabel(self, sample_holder):
        mapper = {'a':'d', 'b':'e', 'c':'f', 'signal':'signal_relabel'}
        sample_holder.relabel(mapper)
        assert list(sample_holder.keys())==list(mapper.values())

    def test_data_presence(self, sample_holder):
        assert 'a' in sample_holder
        assert 'a,b' in sample_holder
        assert ['a', 'b'] in sample_holder
        assert 'z' not in sample_holder
