# Physio Cassette: Storage structures for signals, metadata and event annotations
Managing signals in physiology and associated metadata can be a pain.

You want abstraction, but not much from any underlying Numpy array holding them.

You want annotations in signals, but not obscure representations.

Physio Cassette is just that: Numpy arrays and dictionaries with flair. Physio Cassette provides also automatic caching operations using pickle and matlab storage


### Basic data structures
- Signal: a numpy array with associated sampling frequency, timestamps and minor metadata. Zero-cost abstraction, the data can be accessed directly
- EventRecord: a class for time annotated events based on [traces](https://github.com/datascopeanalytics/traces) TimeSeries with support for binary, trains, and multilevel events
Signals can be iterated using EventRecord events as anchor points and viceversa Events can be converted to a sampled Signal

### Containers
- DataHolder: your box of cables based on Python dictionary. Parent class of SignalFrame and EventFrame.
- SignalFrame: A container for Signal data structures.
- EventFrame: A container for EventRecord structures, with support to merge operations (e.g. events annotated across multiple channels).

### Supported IO
Physio-cassette aims to support seamlessly different file and data formats.
XML format is currently based on NSRR interpretation of data annotations.
Some functionalities will be added in the future. Other format specific features (e.g. physical/digital ranges in EDF and WFDB) are absent on purpose
| Structure   | Numpy arrays       | CSV/columnar files | Matlab files                       | EDF files          | Physionet WFDB     | XML                |
|-------------|--------------------|--------------------|------------------------------------|--------------------|--------------------|--------------------|
| Signal      | :heavy_check_mark: |                    | :heavy_check_mark:                 | (use SignalFrame)  | (use SignalFrame)  |                    |
| SignalFrame | :heavy_check_mark: |                    | :heavy_check_mark: (1 file/signal) | :heavy_check_mark: | :heavy_check_mark: |                    |
| EventRecord | :heavy_check_mark: | :heavy_check_mark: |                                    |                    | :heavy_check_mark: | :heavy_check_mark: |
| EventFrame  | (use EventRecords) | :heavy_check_mark: |                                    |                    | :heavy_check_mark: | :heavy_check_mark: |

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
- [openpyxl-dictreader](https://pypi.org/project/openpyxl-dictreader/) (excel files IO)
- [xlrd](https://pypi.org/project/xlrd/) (old excel format)
- [pyedflib](https://github.com/holgern/pyedflib)
- [pymatreader](https://pypi.org/project/pymatreader/)
- [dateutil](https://pypi.org/project/python-dateutil/)
- [wfdb](https://pypi.org/project/wfdb/)
- [xmltodict](https://pypi.org/project/xmltodict/)


#### Contributing
Looking for people more experienced in writing unit tests and overall beta-testers to help with the reliability of the library

If you feel generous and this library helped your project:

[![Buy me a coffee][buymeacoffee-shield]][buymeacoffee]

[buymeacoffee]: https://www.buymeacoffee.com/u2Vb3kO
[buymeacoffee-shield]: https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png