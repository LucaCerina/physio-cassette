# Physio Cassette: Storage structures for signals, metadata and event annotations
Managing signals in physiology and associated metadata can be a pain.

You want abstraction, but not much from any underlying Numpy array holding them.

You want annotations in signals, but not obscure representations.

Physio Cassette is just that: Numpy arrays and dictionaries with flair. Physio Cassette provides also automatic caching operations using pickle and matlab storage


### Basic data structures
- Signal: a numpy array with associated sampling frequency, timestamps and minor metadata. Zero-cost abstraction, the data can be accessed directly
- EventRecord: a class for time annotated events based on [traces](https://github.com/datascopeanalytics/traces) TimeSeries with support for binary, trains, and multilevel events

### Containers
- DataHolder: your box of cables based on Python dictionary. Parent class of SignalFrame and EventFrame.
- SignalFrame: A container for Signal data structures. Can load data from numpy arrays, MatLab .mat files and EDF files
- EventFrame: A container for EventRecord structures, with support to merge operations (e.g. events annotated across multiple channels).
Can load data from CSV files, with each annotated label stored as its own EventRecord.

### Caching
To cache an operation simply do:
```python
from physio_cassette import autocache

def your_function(x:int) -> bool:
    # Some long operation you want to cache
    return True

result = autocache(your_function, '~/path_to_cache_folder', 'desired_cache_file')(1)
```

## Installation
To install PhysioCassette run:
```bash
$ pip install physio_cassette
```

## Dependencies
- Numpy
- Scipy (Matlab IO)
- [traces](https://github.com/datascopeanalytics/traces)
- [pyedflib](https://github.com/holgern/pyedflib)
- [pymatreader](https://pypi.org/project/pymatreader/)


#### Contribution
If you feel generous and this library helped your project:

[![Buy me a coffee][buymeacoffee-shield]][buymeacoffee]

[buymeacoffee]: https://www.buymeacoffee.com/u2Vb3kO
[buymeacoffee-shield]: https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png