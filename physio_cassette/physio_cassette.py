from __future__ import annotations

import importlib
import re

# -*- coding: utf-8 -*-

"""
This module implements simple abstractions over data and I/O to store signals, time series, metadata and event annotations.
Designed for physiological signals, but general enough to be used with any type of data.
"""

import csv
import inspect
import os
import pickle
import warnings
from collections import namedtuple
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from glob import glob
from hashlib import blake2b
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Number
from operator import *
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Tuple, Union
from types import MappingProxyType

import numpy as np
import openpyxl_dictreader
import pyedflib as edf
import wfdb
import xlrd
import xmltodict
from dateutil import parser
from pymatreader import read_mat
from scipy.io import savemat
from traces import TimeSeries

__all__ = [
    'XLSDictReader',
    'DataHolder',
    'SignalbyEvent',
    'Signal',
    'SignalFrame',
    'EventRecord',
    'EventFrame',
    'autocache'
]

# Default matlab format for signals
MATLAB_DEFAULT_SIGNAL_FORMAT = {
    'data': 'data',
    'sampling_rate': 'SampleRate',
    'start_date': 'StartDate',
    'start_time': 'StartTime'
}

def parse_timestamp(timestamp:str, start_time:datetime) -> datetime:
    """Parse a timestamp string to add it to an EventRecord/Frame.
    A timestamp is always assumed to be after the start time

    Args:
        timestamp (str): String to be parsed
        start_time (datetime): Start time of the recording

    Returns:
        datetime: Parsed datetime
    """
    if isinstance(timestamp, datetime):
        warnings.warn("The input is already a datetime object, returning input")
        return timestamp
    if len(timestamp)<3:
        # Unlikely to be any reasonable format
        return None
    # Initial assumption is timestamp has ISO8601 format
    try:
        output = datetime.fromisoformat(timestamp)
    except ValueError:
        if len(timestamp)>10:
            try:
                output = parser.parse(timestamp, fuzzy=True, default=datetime.fromtimestamp(0))
            except Exception:
                # TODO extend functionalities to support other formats
                raise ValueError("Unsupported data format. Update your data")
        else:
            timestamp = timestamp.replace('.',':', 2)
            output = datetime.strptime(timestamp, "%H:%M:%S")

    # Correct for start_time
    if output < start_time:
        output = output.replace(year=start_time.year, month=start_time.month, day=start_time.day)
        if output.hour<12 and start_time.hour>12:
            output += timedelta(days=1)

    return output

class XLSDictReader(object):
    """XLS file reader for old excel format compatibility. Based on https://gist.github.com/mdellavo/639082 and openpyxl_dictreader
    # TODO move it to an external dependency
    """
    def __init__(self, filename:str, fieldnames:list=None, logfile:str=os.devnull) -> None:
        self.wb = xlrd.open_workbook_xls(filename, formatting_info=True, logfile=open(logfile, 'w'))
        self.ws = self.wb.sheet_by_index(0)
        self.reader = self._reader(self.ws)
        self._fieldnames = fieldnames
        self.line_num = 0

    def _reader(self, iterator):
        cvalue = lambda x: x.value if x.ctype != 3 else xlrd.xldate_as_datetime(x.value,0).isoformat()
        total = [[cvalue(col) for col in row] for row in iterator]
        for row in total:
            yield row

    @property
    def fieldnames(self):
        if self._fieldnames is None:
            try:
                self._fieldnames = next(self.reader)
            except StopIteration:
                pass
        self.line_num += 1
        return self._fieldnames

    @fieldnames.setter
    def fieldnames(self, value):
        self._fieldnames = value

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.line_num == 0:
            self.fieldnames
        row = next(self.reader)
        self.line_num += 1

        while all(cell is None for cell in row):
            row = next(self.reader)

        d = dict(zip(self.fieldnames, row))
        lf = len(self.fieldnames)
        lr = len(row)
        if lf < lr:
            d[self.restkey] = row[lf:]
        elif lf > lr:
            for key in self.fieldnames[lr:]:
                d[key] = self.restval
        return d

class DataHolder(dict):
    """Minor abstraction over dict class to hold information together and check labels
    """
    def __init__(self,*arg,**kw):
        super(DataHolder, self).__init__(*arg, **kw)

    def __iter__(self) -> Iterator:
        return super().items().__iter__()

    def relabel(self, mapper:dict):
        """Update labels

        Args:
            mapper (dict): Dictionary with new labels
        """
        for key, val in mapper.items():
            if key in self:
                self[val] = self.pop(key)
                if hasattr(self[val], 'label'):
                    self[val].label = val
    
    def __contains__(self, input_labels:Union[str, Iterable]) -> bool:
        """Check if every label given as input is present in the DataHolder

        Args:
            input_labels (Union[str, Iterable]): string in the format A,B,C or iterable [A,B,C]

        Returns:
            bool: If every label in input_labels is present
        """
        # Distinguish pure iterables from string labels e.g. EEG,ECG,Flow
        if isinstance(input_labels, str):
            labels_list = input_labels.split(',')
        else:
            labels_list = input_labels
        
        return all([x in self.labels for x in labels_list])

    # Labels property
    @property
    def labels(self):
        """Return label names

        Returns:
            [list]: label of all the data in the DataHolder
        """
        return [*self.keys()]

class SignalbyEvent(namedtuple('SignalbyEvent', ['data', 'value', 'timestamp', 'distance', 'overlap'])):
    """An helper namedtuple class to manipulate Signal.sample_by_events return variable

    """
    __slots__ = ()

    @property
    def duration(self):
        return self.distance[1]

@dataclass(repr=False)
class Signal:
    """A class that holds any signal recorded from a sensor
    TODO changes in data or fs may not be reflected in tstamps

    Class Attributes:
        label: str
            label of the specific sensor
        data: np.ndarray
            the signal values
        fs: float
            sampling frequency of the signal
        start_time: datetime
            starting time of the signal
    
    Attributes:
        time: np.ndarray(datetime)
            time instants of each sample
        shape: Tuple(int)
            shape of the data array
    """
    label: str = ''
    data: np.ndarray = None
    fs: float = 1.0
    start_time: datetime = datetime.fromtimestamp(0)
    tstamps: np.ndarray = None

    def __post_init__(self):
        assert self.fs>0 and ~np.isnan(self.fs), f"Sampling frequency should be positive and a valid number. Got {self.fs}"
        self.data = np.array(self.data) if self.data is not None else np.array([])
        self.tstamps = np.array(self.tstamps) if self.tstamps is not None else np.array([])

    def __setitem__(self, indexes:Union[slice,Integral,np.ndarray, list], values:Union[np.ndarray, Number]):
        """Update values of a slice of the Signal

        Args:
            indexes (Union[slice,Integral,np.ndarray, list]): Index of the item/s to be set : slice example sig[1:1000], integer, array or list
            values (Union[np.ndarray, Number]): new values
        """
        if not (isinstance(indexes, (slice, Integral)) or (isinstance(indexes, (np.ndarray, list))\
               and any([np.issubdtype(indexes[0], np.bool_), np.issubdtype(indexes[0], np.integer)]))):
            raise IndexError("only integers, slices (`:`) and integer or boolean arrays are valid indices")
        assert isinstance(values, np.ndarray) or isinstance(values, Number)
        self.data[indexes] = values

    def update(self, **kwargs) -> Signal:
        """Modify a Signal and return a new instance. For example sig.update(data=[...]) would change the data

        Returns:
            Signal: Updated signal
        """
        origin_dict = self.__dict__
        # Checks on the input
        assert set(kwargs.keys()).issubset(set(origin_dict)), f"Some arguments of Signal update are invalid: {set(kwargs.keys()).difference(set(origin_dict))}"
        
        # Updated data, tstamps are always reset (checking them would have a lot of edge cases)
        output_dict = dict(origin_dict, **kwargs)
        output_dict['tstamps'] = None
        
        return Signal(**output_dict)

    def sub(self, indexes:Union[slice,Iterable]=None, stop:Any=None, start:Any=None, step:Any=None, reset_start_time:bool=True):
        """Return a slice of self as a Signal object

        Args:
            indexes (slice, Iterable): example slice(None,10,None), slice (1, 10, None), [1,10,2]. Single values are not accepted, use __getitem__ slicing

        Returns:
            [Signal]: sliced signal
        """
        assert ((stop is not None) or (start is not None)) ^ (indexes is not None), "At least indexes, stop, or start and stop should be assigned, but not together"
        assert ~np.issubdtype(type(indexes), np.integer), "Please refrain from calling sub on a single value, use your_signal[idx] instead"
        assert (indexes is None) or (isinstance(indexes, slice) or (isinstance(indexes, Iterable) and len(indexes)<=3)), "Invalid indexes, use slice object or a iterable with a maximum length of 3"

        # Define indexes
        if indexes is not None:
            _indexes = indexes if isinstance(indexes, slice) else slice(*indexes)
        else:
            _start = 0 if start is None else start
            _indexes = slice(_start, stop, step)

        # Check for start time slicing
        if reset_start_time:
            _start_time = self.tstamps[_indexes.start] if ((self.tstamps is not None) and len(self.tstamps)) else self.start_time + timedelta(seconds=_indexes.start/self.fs)
        else:
            _start_time = self.start_time

        return Signal(
            label=self.label,
            data=self.data[_indexes],
            fs=self.fs,
            start_time=_start_time,
            tstamps=self.tstamps[_indexes] if self.tstamps is not None else self.tstamps
        )
    
    def sample_by_events(self, events:EventRecord, start_time:datetime=None, target_values:Any=None, window_length:Union[float, Tuple]=30, direction:str='both', ignore_first_event:bool=True) -> Iterator[Tuple[np.ndarray, Any, datetime, Tuple, Tuple]]:
        """Iterate segments of a Signal using events in EventRecord as anchor points. Select events of interest and configure width of samples around the event.
           Each sample returns the timing and value of the event, the distance with adjacent events (NaN for first and last event), and if adjacent events overlap with sample length.

        Args:
            events (EventRecord): EventRecord to be iterated
            start_time (datetime, optional): Override start time if signal and events are not aligned. Defaults to None.
            target_values (Any, optional): Which target values in events should yield a sample. Defaults to None, all values are used.
            window_length (Union[float, Tuple], optional): Length of the window to be used. Can be two values for asymmmetric windows. Defaults to 30s.
            direction (str, optional): Direction of the samples, can be up to the event ('backward'), after it ('forward') or around it ('both'). Defaults to 'both'.
            ignore_first_event (bool, optional): Ignore the first value, even if in target (e.g. binary records starting with 0). Defaults to True.

        Yields:
            np.ndarray: signal sample
            Any: value of the event
            datetime: datetime of the event
            Tuple: distance with adjacent events
            Tuple: True if adjacent events overlap with the window
        """
        # Asserts
        assert isinstance(events, EventRecord), "The events keyword expects an EventRecord variable"
        assert direction in ['both', 'backward', 'forward'], f"Valid sampling directions are 'both', 'backward', 'forward', got {direction}"
        assert (start_time is not None) or (events.start_time==self.start_time), "Specify start time if the signal and events have different start times"
        assert (isinstance(window_length, (float,int)) and window_length>0) or (isinstance(window_length, tuple) and len(window_length)==2 and all([x>0 for x in window_length])), "window_length accepts only one or two numbers, positive"
        _start_time = self.start_time if start_time is None else start_time

        # Event checker lambdas
        is_target = lambda x: True if target_values is None else (x in target_values if isinstance(target_values, Iterable) else x==target_values)
        event_distance = lambda i,j: (events.data.get_item_by_index(j)[0] - events.data.get_item_by_index(i)[0]).total_seconds()

        # Sample spacing
        _window_length = window_length if isinstance(window_length, tuple) else (window_length, window_length)
        back_samples = int(self.fs*_window_length[0]) if direction in ['both', 'backward'] else 0
        fwd_samples = int(self.fs*_window_length[1]) if direction in ['both', 'forward'] else 0

        # Iterator
        for i, (ts, val) in enumerate(events.data.items()):
            if i==0 and ignore_first_event:
                continue
            if is_target(val):
                # Slice block of the signal
                event_sample = int((ts-_start_time).total_seconds()*self.fs)
                signal_slice = slice(event_sample-back_samples, event_sample+fwd_samples)
                # Distance with adjacent events
                back_event_distance = event_distance(i-1, i) if i>0 else np.nan
                fwd_event_distance = event_distance(i, i+1) if i+1<events.data.n_measurements() else np.nan
                event_overlap = (back_event_distance<_window_length[0], fwd_event_distance<_window_length[1])
                if signal_slice.start>=0 and signal_slice.stop<self.shape[0]:
                    yield SignalbyEvent(self[signal_slice], val, ts, (back_event_distance, fwd_event_distance), event_overlap)

    def __getitem__(self, indexes):
        """Return only a data slice, without other Signal attributed attached

        Args:
            indexes (slice, int): Example sig[1:1000], sig[5], sig[3:]

        Returns:
            [np.ndarray]: sliced data
        """
        return self.data[indexes]

    def __array__(self, dtype=None):
        return self.data

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"Signal: {self.data.__repr__()}, fs={self.fs}, start_time={self.start_time.isoformat()}"

    def __unary_op__(self:Signal, other:Union[int, float, np.ndarray, Signal], op:function, r:bool=False) -> Signal:
        """Apply a unary operation, accounting for edge cases

        Args:
            self (Signal): Input Signal
            other (Union[int, float, np.ndarray, Signal]): other operand
            op (function): operation
            r (bool, optional): operation is reversed. Defaults to False.

        Returns:
            Signal: Result of the operation
        """
        output = Signal(**vars(self))
        if np.issubdtype(type(other), np.number):
            output.data = op(self.data, other) if (not r) else op(other, self.data)
        elif isinstance(other, np.ndarray):
            if len(self)==len(other) and self.ndim==other.ndim:
                output.data = op(self.data, other) if (not r) else op(other, self.data)
            else:
                raise ValueError(f"The Signal object and the array should have same size and dimensions, got {self.shape} and {other.shape}")
        elif isinstance(other, Signal):
            if self.fs==other.fs:
                overlap_start, self_slice, other_slice = self.__get_overlap__(other)
                output.data = []
                if overlap_start is not None:
                    output.data = op(self[self_slice],other[other_slice]) if (not r) else op(other[self_slice], self[self_slice])
                    output.start_time = overlap_start
                    output.tstamps = None
            else:
                raise ValueError(f"The two Signal objects should have the same sampling frequency, got {self.fs} and {other.fs}")
        else:
            return NotImplemented
        
        return output

    def __mul__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, mul)

    def __rmul__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, mul, True)
    
    def __truediv__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, truediv)
    
    def __rtruediv__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, truediv, True)

    def __add__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, add)

    def __radd__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, add, True)

    def __sub__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, sub)

    def __rsub__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, sub, True)
    
    def __pow__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, pow)
    
    def __rpow__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, pow, True)

    def __logical_op__(self:Signal, other:Union[int, float, bool], op:function) -> np.ndarray:
        """Apply logical operation only on data. DOES NOT return a Signal

        Args:
            self (Signal): Input signal
            other (Union[int, float, bool]): Scalar comparison value
            op (function): logical operator

        Returns:
            np.ndarray: Output of the logical operation
        """
        if np.issubdtype(type(other), np.number) or np.issubdtype(type(other), bool):
            return op(self.data, other)
        else:
            return NotImplemented

    def __lt__(self:Signal, other:Union[int,float,bool]) -> np.ndarray:
        return self.__logical_op__(other, lt)

    def __le__(self:Signal, other:Union[int,float,bool]) -> np.ndarray:
        return self.__logical_op__(other, le)

    def __eq__(self:Signal, other:Union[int,float,bool]) -> np.ndarray:
        return self.__logical_op__(other, eq)

    def __ne__(self:Signal, other:Union[int,float,bool]) -> np.ndarray:
        return self.__logical_op__(other, ne)

    def __ge__(self:Signal, other:Union[int,float,bool]) -> np.ndarray:
        return self.__logical_op__(other, ge)

    def __gt__(self:Signal, other:Union[int,float,bool]) -> np.ndarray:
        return self.__logical_op__(other, gt)

    def __get_overlap__(self:Signal, other:Signal) -> Tuple[datetime, slice, slice]:
        """Get overlap between two Signal objects with same sampling frequency. Utility for unary operations

        Args:
            self (Signal): One Signal
            other (Signal): Other Signal

        Returns:
            Tuple[datetime, slice, slice]:
            Overlapping start_time, or None if it does not exist
            Valid slice of self, Valid slice of other
        """
        assert self.fs==other.fs, "Overlap currently not defined for signals with different sampling frequency"
        if (self.start_time != other.start_time) or len(self)!=len(other):
            warnings.warn("Different start times or length. Only overlapping samples will be stored")

        # Get larger start time
        start_time = max(self.start_time, other.start_time)
        if (start_time > self.start_time+timedelta(seconds=len(self)*self.fs)) or (start_time > other.start_time+timedelta(seconds=len(other)*other.fs)):
            warnings.warn("No overlap between the data")
            start_time = None
            self_slice = slice(0,0)
            other_slice = slice(0,0)
        else:
            self_start = int((start_time-self.start_time).total_seconds()//self.fs)
            other_start = int((start_time-other.start_time).total_seconds()//other.fs)
            # Get overlapping samples
            min_common_samples = min(len(self[self_start:]), len(other[other_start:]))
            self_stop = self_start+min_common_samples
            other_stop = other_start+min_common_samples
            self_slice = slice(self_start,self_stop)
            other_slice = slice(other_start,other_stop)
        
        return start_time, self_slice, other_slice

    @classmethod
    def from_mat_file(cls, mat_filename:str, data_format:dict=MATLAB_DEFAULT_SIGNAL_FORMAT, time_format:str="%d/%m/%Y-%H:%M:%S") -> Tuple[Any, str]:
        """Load a Signal from formatted matlab file containing the data, start date and time, and sampling rate.

        Args:
            mat_filename (str): input filename
            data_format (dict, optional): Name of the variables in the mat file. Defaults to MATLAB_DEFAULT_SIGNAL_FORMAT.
            time_format (str, optional): format used to parse date and time string. Defaults to "%d/%m/%Y-%H:%M:%S".

        Returns:
            Tuple[Any, str]: the loaded `py:class:~Signal` (None if it cannot be loaded), label of the signal

        Raises:
            ValueError if Start datetime variables are missing in the matlab file
        """

        assert Path(mat_filename).suffix == '.mat', "Wrong file format, expected a Matlab file ending with .mat suffix"
        label = Path(mat_filename).stem

        try:
            raw_mat = read_mat(mat_filename, variable_names=list(data_format.values()))
            assert data_format['start_date'] in raw_mat.keys(), "start date missing in Mat data file"
            assert data_format['start_time'] in raw_mat.keys(), "start time missing in Mat data file"
        except (ValueError, AssertionError) as e:
            warnings.warn(f"{e} {mat_filename}")
            return None, label

        start_time = datetime.strptime(f"{raw_mat[data_format['start_date']]}-{raw_mat[data_format['start_time']]}", time_format)
        return Signal(label=label, data=raw_mat[data_format['data']], fs=raw_mat[data_format['sampling_rate']], start_time=start_time), label

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def time(self):
        """Return time samples, calculate and store them if it wasn't done before

        Returns:
            [np.ndarray]: time instants of each sample
        """
        if self.tstamps is None or self.tstamps.shape[0] != self.data.shape[0]:
            if not np.isnan(self.fs):
                steps = np.linspace(0, (self.data.shape[0]-1)/self.fs, self.data.shape[0])
                tsteps = [self.start_time + timedelta(seconds=x) for x in steps]
            else: # TODO strong assumption that it's a signal determined by events arrival (e.g. RR intervals)
                tsteps = [self.start_time + timedelta(seconds=x) for x in np.cumsum(self.data)]
            self.tstamps = np.array(tsteps)

        return self.tstamps

    @property
    def shape(self):
        return self.data.shape
    

class SignalFrame(DataHolder):
    """A class to hold various `py:class:~Signal` together
    
    Attributes:
        start_date: datetime
            start datetime, considered equal for all Signals
        labels: list[str]
            list of labels for all Signals
        dict object: dict
            Signals are stored as a dict with label:Signal format
    """
    start_date = None

    def __init__(self, *args, **kwargs):
        self.start_date = kwargs.pop('start_date', None)
        super(SignalFrame, self).__init__(*args, **kwargs)

    @classmethod
    def from_arrays(cls, labels:Iterable, signals:Iterable, samplings:Iterable, start_time:datetime=datetime.fromtimestamp(0)):
        """Generate a SignalFrame from multiple arrays

        Args:
            labels (Iterable): label for each Signal
            signals (Iterable): data for each Signal
            samplings (Iterable): sampling frequency for each Signal
            start_time (datetime, optional): initial datetime. Defaults to datetime.fromtimestamp(0).

        Returns:
            [self]: initialized SignalFrame
        """
        output = cls(start_date=start_time)
        for label, signal, fs in zip(labels,signals,samplings):
            output[label] = Signal(label=label, data=signal, fs=fs)
        return output
    
    @classmethod
    def from_edf_file(cls, record_filepath:str, signal_names:Union[str,list]=None):
        """Generate a SignalFrame from a single EDF/BDF file

        Args:
            record_filepath (str): filename of the EDF file
            signal_names (Union[str,list], optional): restrict loading to certain signals only, otherwise load everything. Defaults to None.

        Returns:
            [self]: initialized SignalFrame
        """

        # Read edf file
        signals, signal_headers, header = edf.highlevel.read_edf(record_filepath, ch_names=signal_names)
        output = cls(start_date = header['startdate'])
        for sig_header, signal in zip(signal_headers, signals):
            label = sig_header['label']
            if (signal_names is None) or (label in signal_names):
                output[label] = Signal(label=label, data=signal, fs=sig_header['sample_rate'], start_time=output.start_date)
        return output

    @classmethod
    def from_wfdb_record(cls, record_filepath:str, signal_names:Union[str,list]=None):
        """Generate a SignalFrame from a Physionet WFDB record

        Args:
            record_filepath (str): Path to the WFDB record folder with record name repeated e.g. sample_data/tr100/tr100
            signal_names (Union[str,list], optional): restrict loading to certain signals only, otherwise load everything. Defaults to None.

        Returns:
            [self]: initialized SignalFrame
        """
        # Read header
        header = wfdb.rdheader(record_filepath)
        fs = header.fs
        channels = header.sig_name
        start_time = header.base_datetime if header.base_datetime is not None else datetime.fromtimestamp(0)
        output = cls(start_date = start_time)
        # Select channels to be read
        channels_list = channels if signal_names is None else list(set(channels).intersection(set(signal_names)))
        if len(channels_list)==0:
            warnings.warn(f"Selected channels {signal_names} are not available in set {channels}. Returning empty frame!")
            return output

        # Read record
        record = wfdb.rdrecord(record_filepath, channel_names=channels_list)
        for i, ch in enumerate(channels_list):
            output[ch] = Signal(label=ch, data=record.p_signal[:,i], fs=fs, start_time=start_time)
        return output

    @classmethod
    def from_mat_folder(cls, folder:str, signal_names:Union[str,list]=None, data_format:dict=MATLAB_DEFAULT_SIGNAL_FORMAT, time_format:str="%d/%m/%Y-%H:%M:%S"):
        """Generate a SignalFrame from a folder of matlab files, according to a certain format

        Args:
            folder (str): name of the folder
            signal_names (Union[str,list], optional): restrict loading to certain signals only, otherwise load everything. Defaults to everything.
            data_format (dict, optional): Name of the variables in the mat files. Defaults to MATLAB_DEFAULT_SIGNAL_FORMAT.
            time_format (str, optional): format used to parse date and time string. Defaults to "%d/%m/%Y-%H:%M:%S".

        Returns:
            [self]: initialized SignalFrame
        """
        output = cls()
        filenames = glob(folder+'/*.mat')
        for filename in filenames:
            label = Path(filename).stem
            if (signal_names is None) or (label in signal_names):
                signal, label = Signal.from_mat_file(mat_filename=filename, data_format=data_format, time_format=time_format)
                if signal:
                    output[label] = signal
                    # Start date is assumed to be common for all signals, otherwise consider the oldest one
                    if (output.start_date is None) or (signal.start_time < output.start_date): 
                        output.start_date = signal.start_time
        return output

    def sample_by_events(self, labels:Union[str,list], events:EventRecord, start_time:datetime=None, target_values:Any=None, window_length:Union[float, Tuple]=30, direction:str='both', ignore_first_event:bool=True) -> Iterator[Tuple[MappingProxyType, Any, datetime, Tuple, Tuple]]:
        """Iterate segments of a SignalFrame using events in EventRecord as anchor points. Select events of interest and configure width of samples around the event.
           Each sample returns the timing and value of the event, the distance with adjacent events (NaN for first and last event), and if adjacent events overlap with sample length.
           Data from signals is returned in a NamedTuple.
           Note! the samples will have different sizes if the sampling rate is different. Sampling rates are not returned in the current implementation

        Args:
            labels (Union[str,list], optional): Which signals must be returned
            events (EventRecord): EventRecord to be iterated
            start_time (datetime, optional): Override start time if signal and events are not aligned. Defaults to None.
            target_values (Any, optional): Which target values in events should yield a sample. Defaults to None, all values are used.
            window_length (Union[float, Tuple], optional): Length of the window to be used. Can be two values for asymmmetric windows. Defaults to 30s.
            direction (str, optional): Direction of the samples, can be up to the event ('backward'), after it ('forward') or around it ('both'). Defaults to 'both'.
            ignore_first_event (bool, optional): Ignore the first value, even if in target (e.g. binary records starting with 0). Defaults to True.

        Yields:
            MappingProxyType(str,np.ndarray): immutable dict with keys as the labels in the data and samples as values
            Any: value of the event. Constant for all labels
            datetime: datetime of the event. Constant for all labels
            Tuple: distance with adjacent events. Constant for all labels
            Tuple: True if adjacent events overlap with the window. Constant for all labels
        """
        # Check available labels and instantiate iterators
        _labels = [x for x in labels] if isinstance(labels, list) else [labels]
        _iterators = []
        for label in _labels:
            if label not in self:
                warnings.warn(f"Label {label} missing in the SignalFrame. It will be ignored.")
                _labels.remove(label)
                continue
            iterator = self[label].sample_by_events(events, start_time, target_values, window_length, direction, ignore_first_event)
            _iterators.append(iterator)
        assert len(_labels)>0, f"The SignalFrame doesn't contain any of the labels {labels}"

        # Iterate and yield
        for samples in zip(*_iterators):
            # Get data
            output_samples = {}
            # TODO check what happens with Signals of different lengths
            for k, sample in zip(_labels, samples):
                output_samples[k] = sample.data
            output_samples = MappingProxyType(output_samples)

            yield SignalbyEvent(output_samples, sample.value, sample.timestamp, sample.distance, sample.overlap)

@dataclass
class EventRecord:
    """A class describing events sampled at irregular times
    
    Attributes:
        label: str
            label of the data in the record
        start_date: datetime
            start datetime
        data: traces.TimeSeries
            timeseries with records at irregular timestamps
        is_binary: bool
            flag variable if the TimeSeries has only two possible states
        is_spikes: bool
            flag variable if the TimeSeries encodes events (with null/negligible duration) instead of states/levels
        start_value: Any
            starting value at start_time
    """
    label: str = ''
    start_time: datetime = datetime.fromtimestamp(0)
    data: TimeSeries = field(default_factory=TimeSeries)
    is_binary: bool = False
    is_spikes: bool = False
    start_value: Any = None

    def __post_init__(self):
        """Assign start value if not present already. Allow for other future checks

        Returns:
            [self]: initialized EventRecord
        """
        if not isinstance(self.data, TimeSeries):
            self.data = TimeSeries(self.data, default=self.start_value)
        if self.start_value is not None and len(self.data) and self.data.first_key() > self.start_time:
            self.data[self.start_time] = self.start_value
        return self

    def __len__(self):
        """Override __len__ to return the number of events.
        In binary series first 'non-event' is subtracted. Start and end of event (state transitions) are considered together, so divided by 2
        In non binary series returns number of measurements or 0.

        Returns:
            int: Number of events
        """
        if not self.is_binary:
            return self.data.n_measurements()
        elif self.data.n_measurements()>1:
            remainder = self.data.n_measurements() % 2 # Assess that the start/0 value may be overwritten
            return (self.data.n_measurements()-remainder)//2 
        else:
            return 0 # Assuming a binary EventRecord with only the start/0 value to be empty

    def __index_to_key(self, start:int=None, stop:int=None) -> Tuple[Any,Any]:
        """Convert integer indexes to TimeSeries keys

        Args:
            start (int, optional): initial index. Defaults to None.
            stop (int, optional): end index. Defaults to None.

        Returns:
            Tuple[Any,Any]: keys associates with input indexes
        """
        tstart = self.data.get_item_by_index(start)[0] if start is not None else self.data.first_key()
        tend =  self.data.get_item_by_index(stop)[0] if stop is not None else self.data.last_key()
        return tstart, tend

    def __min_event_interval(self) -> float:
        """Return minimum time interval in seconds in the data

        Returns:
            [float]: Minimum time interval
        """
        min_interval = np.inf
        for curr,next in self.data.iterintervals():
            interval = (next[0]-curr[0]).total_seconds()
            min_interval = interval if interval<=min_interval else min_interval
        return min_interval

    def __getitem__(self, indexes):
        """Return a sliced EventRecord or a single value

        Args:
            indexes (slice, int): slice or single index, example rec[1:1000] or rec[5]

        Returns:
            [EventRecord]: Sliced instance
        """
        if not isinstance(indexes, slice):
            if np.issubdtype(type(indexes), np.integer):
                return self.data.get_item_by_index(indexes)[1]
            else:
                return self.data[indexes]

        if indexes.start==indexes.stop==indexes.step==None:
            return self

        if isinstance(indexes.start, int) or isinstance(indexes.stop, int):
            tstart, tend = self.__index_to_key(indexes.start, indexes.stop)
        else:
            tstart = indexes.start if indexes.start is not None else self.data.first_key()
            tend = indexes.stop if indexes.stop is not None else self.data.last_key()

        # Do not interpolate content if it is a spiking record
        if self.is_spikes:
            output_data = TimeSeries(default=self.data.default, data={k:v for k,_,v in self.data.iterperiods(tstart, tend, lambda t1,t2,v: t1 in self.data._d)})
            tstart = output_data.first_key() if len(output_data) else tstart
        else:
            output_data = self.data.slice(tstart, tend)

        return EventRecord(label=self.label, start_time=tstart, data=output_data, is_binary=self.is_binary, start_value=self.start_value)
    
    def __setitem__(self, index:datetime, value:Any):
        assert isinstance(index, (datetime, slice)), f"EventRecord expects a datetime or slice of datetimes index to set items. Got {type(index)}"
        self.data.__setitem__(index, value)
    
    def __iter__(self):
        return iter(self.data)
    
    def __repr__(self):
        return f"EventRecord \"{self.label}\": {'spiking,' if self.is_spikes else ''}{'binary,' if self.is_binary else ''}start_time {self.start_time}. Data:\n{self.data.__repr__()}"

    @classmethod
    def from_logical_array(cls, label:str, t0:datetime, input_array:np.ndarray, fs:float=1):
        """Create an EventRecord from a logical array representing state changes

        Args:
            label (str): label of the data instance
            t0 (datetime): initial datapoint, 
            input_array (np.ndarray): input array 
            fs (float, optional): Sampling frequency of samples in the array. Defaults to 1Hz.

        Returns:
            EventRecord: output EventRecord
        """
        if np.sum(input_array)<1 or np.sum(input_array)>=input_array.shape[0]:
            return cls.from_ts_dur_array(label=label, t0=t0, ts_array=[], duration_array=[], is_binary=True, start_value=0)

        # Convert to integer array
        int_array = np.clip(input_array, 0, 1) if not np.issubdtype(input_array.dtype, bool) else input_array.astype(int)

        # Detect state changes
        starts = np.where(np.diff(int_array)==1)[0]
        ends = np.where(np.diff(int_array)==-1)[0]
        ends = ends if ends.size else [len(int_array)]
        ends = ends if ends[0]>starts[0] else ends[1:]
        starts = starts if len(starts)==len(ends) else starts[:-1]

        # Create ts and duration arrays
        durations = (ends - starts)/fs
        ts = starts/fs

        return cls.from_ts_dur_array(label=label, t0=t0, ts_array=ts, duration_array=durations, is_binary=True, start_value=0)

    @classmethod
    def from_state_array(cls, label:str, t0:datetime, input_array:Iterable, ts_array:Iterable=None, ts_sampling:float=1.0, start_value:Any=None, compact:bool=False):
        """Generate an EventRecord from an array of states, with a fixed sampling time in seconds or a timestamps array.

        Args:
            label (str): label of the data instance
            t0 (datetime): initial timestamp
            input_array (Iterable): list of events
            ts_array (Iterable, optional): list of timestamps (seconds or datetime) for the events. Defaults to None -> Use fixed sampling
            ts_sampling (float, optional): Desired sampling time. Defaults to 1.0 seconds.
            compact (bool, optional): Compact the output Timeseries or not. Defaults to False
            start_value (Any, optional): Initial value at t0. Defaults to None.

        Returns:
            [self]: initialized EventRecord
        """
        if isinstance(input_array, np.ndarray):
            assert input_array.ndim==1, "EventRecord from_state_array accepts only 1D arrays"
        
        if ts_array is not None:
            assert len(input_array)==len(ts_array), "Input and timestamps array should have the same length"

        if len(input_array)==0:
            warnings.warn("Empty input in EventRecord data")
            return cls.from_ts_dur_array(label=label, t0=t0, ts_array=[], duration_array=[], is_binary=True, start_value=0)
        
        # Define if input is binary or spikes
        unique_states = len(set(input_array))
        _is_binary = unique_states==2 or (unique_states<2 and (start_value is not None and start_value!=input_array[0]))
        _is_spikes = unique_states==1 # TODO this leads to bugs

        get_ts = lambda i: t0 + timedelta(seconds=i*ts_sampling) if ts_array is None else (ts_array[i] if isinstance(ts_array[0], (datetime, np.datetime64)) else t0 + timedelta(seconds=ts_array[i]))
        data = TimeSeries()

        # Assess initial value
        if (start_value is not None and ts_array is not None) and t0!=ts_array[0]:
            data[t0] = start_value

        # Fill EventRecord
        for i, val in enumerate(input_array):
            ts = get_ts(i)
            data[ts] = val
        if compact:
            data.compact()
        _start_value = data[t0]

        return cls(label=label, start_time=t0, data=data, is_binary=_is_binary, is_spikes=_is_spikes, start_value=_start_value)

    @classmethod
    def from_ts_dur_array(cls, label:str, t0:datetime, ts_array:Union[list, np.ndarray], duration_array:Union[list, np.ndarray]=None, is_binary:bool=False, is_spikes:bool=False, start_value:Any=None):
        """Generate an EventRecord from two arrays, one with timestamps and one with duration of events. EventRecord may have binary values and a start_value

        Args:
            label (str): label of the data instance
            t0 (datetime): initial datapoint, set to 0 if start_value is None
            ts_array (Union[list, np.ndarray]): timestamp array, can be datetimes or relative to t0
            duration_array (Union[list, np.ndarray]): duration array, assumed in seconds, should have the same length of ts_array. Use None to encode only events timing
            is_binary (bool, optional): state if the Events have only two possible values. Defaults to False.
            is_spikes (bool, optional): state if the Events encodes only events with no known duration. Defaults to False.
            start_value (Any, optional): Initial value at t0. Defaults to None.

        Returns:
            [self]: initialized EventRecord
        """
        # Transform lists to arrays
        _ts_array = np.asarray(ts_array)
        _is_spikes = is_spikes or (duration_array is None)
        if _is_spikes==False: 
            _duration_array = np.asarray(duration_array)
            assert _ts_array.shape[0] == _duration_array.shape[0], f"Timestamps and Duration arrays should have the same size, got {_ts_array.shape} and {_duration_array.shape}"
        else:
            _duration_array = np.zeros((_ts_array.shape[0]))*np.nan


        # Fill values
        data = TimeSeries()
        if not _is_spikes: # Start value is ignored for spiking arrays
            data[t0], start_value = (0,0) if start_value is None else (start_value, start_value)
        start_time = t0

        # Check for empty arrays
        if _ts_array.size == 0:
            return cls(label=label, start_time=t0, data=data, is_binary=is_binary, start_value=start_value)

        # Timestamps are relative to start_time, assuming to be seconds
        rel_ts_flag = not isinstance(_ts_array[0], (datetime, np.datetime64))
        delta_dur_flag = isinstance(_duration_array[0], timedelta)
        if rel_ts_flag: # Ensure correct type for timedelta
            _ts_array = _ts_array.astype(float)
        for ts, dur in filter(lambda x: x[1]>=0 or _is_spikes, zip(_ts_array, _duration_array)):
            t_start = start_time + timedelta(seconds=ts) if rel_ts_flag else ts
            data[t_start] = 1
            if isinstance(dur, timedelta) or ~np.isnan(dur):
                dur_seconds = dur.total_seconds() if delta_dur_flag else dur
                t_end = (start_time + timedelta(seconds=ts+dur_seconds)) if rel_ts_flag else (ts + timedelta(seconds=dur_seconds)) 
                data[t_end] = 0

        return cls(label=label, start_time=t0, data=data, is_binary=is_binary, is_spikes=_is_spikes, start_value=start_value)

    @classmethod
    def from_csv(cls, record_filepath:str, label:str, event_column:str, t0:datetime, start_value:Any=None, ts_column:str=None, ts_is_datetime:bool=True, ts_sampling:float=0,  delimiter:str=',', skiprows:int=0, **kwargs):
        """Instantiate an EventRecord from a CSV file any column-like file (e.g. Excel files with header)

        Args:
            record_filepath (str): Path of the file to be parsed
            label (str): label assigned to the output EventRecord
            event_column (str): label of the event column in the CSV
            t0 (datetime): initial timestamp
            start_value (Any, optional): Initial value of the record. Defaults to None.
            ts_column (str, optional): label of the timestamps column in the CSV. Defaults to None.
            ts_is_datetime (bool, optional): Flag if the ts column is absolute timestamps, or an increasing value from t0. Defaults to True.
            ts_sampling (float, optional): sampling interval if timestamps are relative. Defaults to 0.
            delimiter (str, optional): CSV column delimiter. Defaults to ','.
            skiprows (int, optional): Skip a certain number of rows before the csv reader
            \\**kwargs: keywords arguments passed to csv reader (e.g. skiprows)

        Returns:
            [EventRecord]: EventRecord instance
        """
        assert skiprows>=0, "Skiprows argument should be positive"

        # Detect file type
        file_type = Path(record_filepath).suffix[1:]
        is_text = file_type in ['csv', 'txt']

        # Fill values
        data = TimeSeries()      
        start_time = t0

        if ts_column is None and ts_sampling == 0:
            raise(ValueError("if ts_column is None a valid ts_sampling in seconds should be passed to the function"))

        event_filter = lambda x: event_column in x
        with open(record_filepath, 'r') as csv_file:
            for _ in range(skiprows*int(is_text)):
                csv_file.readline()
            # Select reader based on format
            if is_text:
                reader = csv.DictReader(csv_file, delimiter=delimiter, **kwargs) 
            elif file_type =='xls':
                reader = XLSDictReader(record_filepath, **kwargs)
            else:            
                reader = openpyxl_dictreader.DictReader(record_filepath, **kwargs)
            # Parse rows
            for row in filter(event_filter, reader):
                value = row[event_column]
                ts = parse_timestamp(row[ts_column], start_time) if ts_is_datetime else start_time + timedelta(seconds=(int(row[ts_column])-1)*ts_sampling)
                if ts is not None:
                    data[ts] = value
        return cls(label=label, start_time=t0, data=data, is_binary=False, start_value=start_value)

    @classmethod
    def from_flat_file(cls, record_filepath:str, label:str, parser:Callable, t0:datetime, start_value:Any=None, default_value:Any=None, ts_is_datetime:bool=True, ts_sampling:float=1.0, skiprows:int=0,):
        """Instantiate an EventRecord from a generic text file with one annotation per line using a custom parser function.
           The function should have a signature str -> str|float, Any, float|None expecting a datetime string of an event or its epoch or relative seconds, its value, and if it ends after a duration in seconds.
           Return None to ignore a line.

        Args:
            record_filepath (str): Path to the file to be parsed
            label (str): Label assigned to the output EventRecord
            parser (Callable): Custom line parser (str -> str|float, Any, float|None)
            t0 (datetime): Initial timestamp
            start_value (Any, optional): Override start value of the EventRecord. Defaults to None.
            default_value (Any, optional): Default value to be used if an event as a duration. Defaults to None -> same as start_value.
            ts_is_datetime (bool, optional): Flag if the ts column is absolute timestamps, or an increasing value from t0. Defaults to True.
            ts_sampling (float, optional): Sampling interval if timestamps are relative. Defaults to 1.0 seconds.
            skiprows (int, optional): Skip a certain number of rows before parsing lines

        Returns:
            [EventRecord]: EventRecord instance
        """
        assert isinstance(parser, Callable), "Expected a function to parse annotation lines"

        # Fill values
        data = TimeSeries()
        start_time = t0
        _default_value = start_value if default_value is None else default_value

        # Keep track of unique values
        unique_states = set()
        if start_value is not None:
            unique_states.add(start_value)

        # Parse record
        has_duration = False
        with open(record_filepath, 'r') as rfile:
            for _ in range(skiprows):
                rfile.readline()
            for line in rfile:
                parsed_line = parser(line)
                if parsed_line is not None:
                    (ts, value, duration) = parsed_line
                    ts = parse_timestamp(ts, start_time) if ts_is_datetime else start_time + timedelta(seconds=float(ts)*ts_sampling)
                    data[ts] = value
                    unique_states.add(value)
                    if duration is not None:
                        has_duration = True
                        data[ts+timedelta(seconds=duration)] = _default_value
        is_binary = data.n_measurements()>1 and len(unique_states)==2
        is_spikes = is_binary and (has_duration==False) and len(unique_states)==1

        return cls(label=label, start_time=t0, data=data, is_binary=is_binary, is_spikes=is_spikes, start_value=start_value)

    @classmethod
    def _get_xml_value(cls, element:dict, event_key:str, default_value:int=None, value_function:Callable=None):
        value = None
        if value_function is not None:
            value = value_function(element)
        elif default_value is not None:
            value = default_value
        else:
            value = element.get(event_key, None)
        return value
    
    @classmethod
    def from_xml(cls, record_filepath:str, label:str, event_key:str, target_values:Union[str,list], ts_key:str, t0:datetime, start_value:Any=None, value_function:Callable=None, duration_key:str=None, events_path:list=[], ts_is_datetime:bool=True, ts_sampling:float=1.0):
        """Instantiate an EventRecord from a XML file. Assuming that a certain depth is represented as a list of annotated tags.

        Args:
            record_filepath (str): Path of the file to be parsed
            label (str): Label assigned to the output EventRecord
            event_key (str): Which XML tag is associated to the type of event
            target_values (Union[str,list]): Values to be stored e.g. ['W', 'N1', 'N2'] or a single string for binary events (e.g. 'apnea')
            ts_key (str): Which XML tag is associated to the timing of the event
            t0 (datetime): Initial timestamp
            start_value (Any, optional): Initial value of the record. Defaults to None.
            value_function (Callable, optional): Set values according to elements in the data. Defaults to None (values are explicit from event_key)
            duration_key (str, optional): Which XML tag is associated to the timing of the event. Defaults to None -> spikes for binary events, automatically embedded for other events (e.g. sleep stages).
            events_path (list, optional): How to reach the right depth in the XML. Defaults to [] -> root of the document.
            ts_is_datetime (bool, optional): Flag if the ts column is absolute timestamps, or an increasing value from t0. Defaults to True.
            ts_sampling (float, optional): Sampling interval if timestamps are relative. Defaults to 1.0 seconds.

        Raises:
            KeyError: Raise an error if the events_path is not correct

        Returns:
            [EventRecord]: EventRecord instance
        """
        # Checks
        if not (callable(value_function) or (value_function is None)):
            value_function = None
            warnings.warn(f"Value function in EventRecord.xml should be None or Callable, got {type(value_function)}. Setting to None")

        # Fill values
        data = TimeSeries()
        start_time = t0
        if start_value is not None:
            data[t0] = start_value

        # Load data
        with open(record_filepath, 'rb') as xml_file:
            input_data = xmltodict.parse(xml_file)
            # Access data nest
            for key in events_path:
                if key in input_data:
                    input_data = input_data.get(key)
                else:
                    raise KeyError(f"Nesting key not found in the parsed XML with path {events_path}")
            assert isinstance(input_data, list), f"Expected events to be a list, got {type(input_data)} for path {events_path}"

        # Parse data
        is_binary = (isinstance(target_values, str) or len(target_values)==1) and (value_function is None)
        binary_set_value = 1 if is_binary else None
        for element in input_data:
            if element[event_key] in target_values:
                ts = parse_timestamp(element[ts_key], start_time) if ts_is_datetime else start_time + timedelta(seconds=float(element[ts_key])*ts_sampling)
                data[ts] = cls._get_xml_value(element, event_key, binary_set_value, value_function)
                if duration_key is not None:
                    ts = ts + timedelta(seconds=float(element[duration_key])*ts_sampling)
                    data[ts] = cls._get_xml_value(element, event_key, start_value, None)
        is_spikes = is_binary and (duration_key is None) and (len(data) >= 1)
        return cls(label=label, start_time=t0, data=data, is_binary=is_binary, is_spikes=is_spikes, start_value=start_value)

                
    @classmethod
    def from_wfdb_annotation(cls, record_filepath:str, label:str, target_values:Union[str,list], t0:datetime, start_value:Any=None, extension:str='ann', openclose:Tuple=None):
        """Instantiate an EventRecord from a Physionet WFDB annotation file.
        If only target values are passed as an argument, the EventRecord will contain observed target values as states.
        If openclose is passed, a binary state will be ON if the annotation starts with the opening character, OFF if it ends with closing one. e.g. '(apnea'->1, 'apnea)'->0

        Args:
            record_filepath (str): Path to the WFDB record folder with record name repeated e.g. sample_data/tr100/tr100
            label (str): Label assigned to the output EventRecord
            target_values (Union[str,list]): Values to be stored e.g. ['W', 'N1', 'N2'] or a single string for openclose scenarios e.g. 'apnea'
            t0 (datetime): Initial timestamp
            start_value (Any, optional): Override start value of the EventRecord. Defaults to None.
            extension (str, optional): Extension of the WFDB annotation file. Defaults to 'ann'.
            openclose (Tuple, optional): Tuple with opening and closing characters for events detection. Defaults to None.

        Returns:
            [EventRecord]: EventRecord instance
        """
        assert (openclose is None) or (len(openclose) == 2 and all([len(x)==1 for x in openclose])), "Openclose should be None or have exactly length two with a single character each"
        # Assign target when looking for opening/closing annotations
        is_binary = len(target_values)<=2 or openclose is not None
        if openclose is not None:
            if isinstance(target_values, list):
                target_key = target_values[0]
                warnings.warn(f"Ignoring extra targets in {target_values} with openclose {openclose}. Use EventFrame function for multiple events")
            else:
                target_key = target_values
         
        # Read record
        record = wfdb.rdann(record_filepath, extension)
        fs = record.fs

        data = TimeSeries()
        start_time = t0

        # Iter data
        for i in range(record.sample.shape[0]):
            # Check where the key is in the annotations
            ann_key = record.aux_note[i] if (len(record.symbol[i].strip())==0 or record.symbol[i]=='"') else record.symbol[i]
            ts = start_time + timedelta(seconds=record.sample[i]/fs)
            if (openclose is None) and ann_key in target_values:
                # Store states separately
                data[ts] = ann_key
            elif (openclose is not None):
                # Check if a state is opening or closing
                if ann_key.startswith(openclose[0]) and ann_key[1:]==target_key:
                    data[ts] = 1
                elif ann_key.endswith(openclose[1]) and ann_key[:-1]==target_key:
                    data[ts] = 0
        return cls(label=label, start_time=t0, data=data, is_binary=is_binary, start_value=start_value)

    def as_array(self, sampling_period:float) -> Signal:
        """Return data as a regularly sampled array. Assign time bin in case of spike arrays.

        Args:
            sampling_period (float): sampling period in seconds.

        Returns:
            Signal: Signal with fixed sampling period.
        """
        # Check minimum interval
        min_interval = self.__min_event_interval()
        if sampling_period > min_interval/2:
            warnings.warn(f"Risk of aliasing jitter for {self.label}! Sampling {sampling_period:.2f} seconds. Minimum interval in data {min_interval:.2f} seconds", RuntimeWarning, stacklevel=2)

        # Assign values
        n_samples = int(np.ceil((self.data.last_key() - self.start_time).total_seconds()/sampling_period))+1 if self.data.last_key()>self.start_time else 1
        n_samples -= 1 if (self.start_time+timedelta(seconds=n_samples*sampling_period) > self.data.last_key()) else 0 # Do not extrapolate at the end
        arr_type = object if isinstance(self.data.first_value(), str) else type(self.data.first_value())
        values = np.zeros((n_samples,), dtype=arr_type)
        if self.is_spikes == False:
            # TODO write proper tests for EventRecord sampling
            # Regular data is resampled without binning
            # Check both first and last values that may not be present
            if self.data.first_key() > self.start_time:
                t0_sample = np.clip(int(np.round((self.data.first_key()-self.start_time).total_seconds()/sampling_period)), None, n_samples-1)
                values[0:t0_sample] = self.start_value
            for (t0, val), (t1, _) in self.data.iterintervals():
                t0_sample = np.clip(int(np.round((t0-self.start_time).total_seconds()/sampling_period)), None, n_samples-1)
                t1_sample = np.clip(int(np.round((t1-self.start_time).total_seconds()/sampling_period)), None, n_samples-1)
                values[t0_sample:t1_sample] = val
            if n_samples>0 and t1_sample < n_samples:
                values[t1_sample:] = val
        else:
            # Assign to closest samples
            for t,_ in self.data.items():
                t_sample = np.clip(int(np.round((t-self.start_time).total_seconds()/sampling_period)), None, n_samples-1)
                values[t_sample] = 1

        output_dtype = np.bool_ if self.is_binary else type(self.data.first_value())
        return Signal(label = self.label, data = np.array(values).astype(output_dtype), fs = 1/sampling_period, start_time=self.start_time)

    def state_frequency(self, sampling_period:float=None) -> Union[Any,Signal]:
        """Return frequency (in Hz) of state change, timestamped at each state starting from the second "high" value
           Return a Signal with fixed sampling if sampling period is not None, a EventRecord otherwise

        Args:
            sampling_period (float, optional): Sampling period in seconds to return a Signal. Defaults to None.

        Returns:
            Union[Any,Signal]: EventRecord of change frequency or Signal sampled at a given period
        """
        assert self.is_binary,  "Method currently implemented only for binary EventRecord instances"

        # We hypothesize the EventRecord starts with "low" value
        output_data = TimeSeries()
        for block in (block for i,block in enumerate(self.data.iterintervals(n=4)) if i%2==0):
            output_data[block[-1][0]] = 1 / (block[-1][0]-block[-2][0]).total_seconds()

        output_record = EventRecord(label=f"{self.label}_state_frequency", start_time=self.start_time, data=output_data, is_binary=False, start_value=self.start_value)

        if sampling_period is None:
            return output_record
        else:
            return output_record.as_array(sampling_period=sampling_period)


    def remap(self, mapper:Union[dict, Callable], default:Any=None) -> None:
        """Update values of events inplace with a dictionary mapping or a function.
        Ignore values that are not in the dictionary mapping or apply default value.
        NOTE casting is done on data to respect mapper keys' type. The casting may raise an error

        Args:
            mapper (Union[dict, Callable]): Mapping dictionary or Callable/lambda
            default (Any, optional): Default value if a value is not available. Defaults to None.

        """
        assert isinstance(mapper, dict) or isinstance(mapper, Callable), "Mapping should be a dict or a function"
        if isinstance(mapper, dict):
            # Cast data according to mapper keys' type
            cast = list(map(type,mapper))[0]
            for t, val in self.data:
                new_val = mapper.get(cast(val), default)
                self.data[t] = val if new_val is None else new_val
            new_start_value = mapper.get(self.start_value)
            self.start_value = self.start_value if new_start_value is None else new_start_value
        else:
            self.data = self.data.operation(None, lambda x,y: mapper(x))
            self.start_value = mapper(self.start_value)

    def binarize(self, key:str, compact:bool=True):
        """Remap helper, returns a binary EventRecord with only 1 where a key is present, 0 elsewhere

        Args:
            key (str): key used as '1' value to binarize the record
            compact (bool, optional): Remove repeated values. Defaults to True.

        Returns:
            EventRecord: binarized EventRecord
        """
        assert type(key)==type(self.data.first_value()), f"Binarize key has type {type(key)}, but data is of type {type(self.data.first_value())}"
        output = copy(self)
        output.remap(lambda x: int(x==key))
        output.is_binary = True
        if compact:
            output.data.compact()

        return output

    @property
    def n_events(self) -> int:
        return len(self)

    @property
    def duration(self) -> float:
        """Return duration of the Record in seconds, as the difference between first key and end.
        Will return 0 if the Record is empty

        Returns:
            float: duration in seconds
        """
        return (self.data.last_key() - self.data.first_key()).total_seconds()

    def time(self, relative:bool=False) -> np.ndarray:
        """Return timestamps of events. Default to absolute timestamps, otherwise relative to start time (in seconds).

        Args:
            relative (bool, optional): Return timestamps as relative to start time. Defaults to False.

        Returns:
            np.ndarray: timestamps array
        """
        tstamps = [x[0] for x in self.data]
        if relative:
            tstamps = [(x-self.start_time).total_seconds() for x in tstamps]
        output = np.array([tstamps]).T
        output.reshape((-1,))
        return output

class EventFrame(DataHolder):
    """A class to hold various EventRecord together
    
    Attributes:
        start_date: datetime
            start datetime, considered equal for all EventRecords
        labels: list[str]
            list of labels for all Records
        dict object: dict
            Records are stored as a dict with label:EventRecord format
    """
    start_date = None

    def __init__(self, *args, **kwargs):
        self.start_date = kwargs.pop('start_date', None)
        super(EventFrame, self).__init__(*args, **kwargs)

    @classmethod
    def from_csv(cls, record_filepath:str, labels: Iterable, event_column: str, ts_column:str, duration_column:str=None, start_time:datetime=datetime.fromtimestamp(0), ts_is_datetime:bool=False, duration_modifier:Callable=None, delimiter:str=',', skiprows:int=0, **kwargs):
        """Instantiate an EventFrame from a CSV file or any column-like file (e.g. Excel files with header), recording certain labels separately

        Args:
            record_filepath (str): Path of the file to be parsed
            labels (Iterable): desired labels to be extracted in the CSV file
            event_column (str): label of the event column
            ts_column (str): label of the timestamp column
            duration_column (str): label of the duration column. If None store as spike EventRecord
            start_time (datetime, optional): Initial datetime of the data. Defaults to datetime.fromtimestamp(0).
            ts_is_datetime (bool, optional): Flag to parse ts column as datetime and not t-t0. Defaults to False.
            duration_modifier (Callable, optional): Alter duration with a custom rule. Defaults to None
            delimiter (str, optional): CSV delimiter. Defaults to ','.
            skiprows (int, optional): Skip a certain number of rows before the csv reader
            \\**kwargs: keywords arguments passed to csv reader (e.g. skiprows)

        Returns:
            [EventFrame]: EventFrame instance
        """
        assert skiprows>=0, "Skiprows argument should be positive"
        assert duration_modifier is None or isinstance(duration_modifier, Callable), "The duration modifier should be a callable function/lambda or None"

        # Detect file type
        file_type = Path(record_filepath).suffix[1:]
        is_text = file_type in ['csv', 'txt']

        start_date = start_time
        # Allocate temporary dictionary
        temp_dict = {}
        for label in labels:
            temp_dict[label] = {'ts':[], 'dur':[]}
        # Parse CSV file
        event_filter = lambda x: event_column in x
        with open(record_filepath, 'r') as csv_file:
            for _ in range(skiprows*int(is_text)):
                csv_file.readline()
            # Select reader based on format
            if is_text:
                reader = csv.DictReader(csv_file, delimiter=delimiter, **kwargs) 
            elif file_type=='xls':
                reader = XLSDictReader(record_filepath, **kwargs)
            else:            
                reader = openpyxl_dictreader.DictReader(record_filepath, **kwargs)
            # Parse rows
            for row in filter(event_filter, reader):
                if any([re.fullmatch(x, row[event_column]) for x in labels]):
                    key = row[event_column]
                    # Add key in temp_dict if not present
                    if key not in temp_dict:
                        temp_dict[key] = {'ts':[], 'dur':[]}
                    # Extract timestamps and duration
                    ts = parse_timestamp(row[ts_column], start_date) if ts_is_datetime else float(row[ts_column])
                    if ts is not None:
                        temp_dict[key]['ts'].append(ts)
                        if duration_column is not None:
                            try:
                                dur = float(row[duration_column])
                            except ValueError:
                                dur = 0
                            temp_dict[key]['dur'].append(dur)
        # Alter duration if necessary
        if duration_modifier is not None:
            for key in temp_dict.keys():
                temp_dict[key]['dur'] = [duration_modifier(x) for x in temp_dict[key]['dur']]
        # Allocate the frames
        output = cls(start_date=start_date)
        for key, data in temp_dict.items():
            _is_spikes = (len(data['dur']) == 0) ^ (len(data['ts']) == 0)
            output[key] = EventRecord.from_ts_dur_array(label=key, t0=start_date, ts_array=data['ts'], duration_array=data['dur'], is_binary=True, is_spikes=_is_spikes)
        return output

    @classmethod
    def from_xml(cls, record_filepath:str, event_key:str, target_values:dict, ts_key:str, t0:datetime, start_value:Any=None, value_function:Callable=None, duration_key:str=None, events_path:list=[], ts_is_datetime:bool=True, ts_sampling:float=1.0):
        """Instantiate an EventFrame from an XML annotation file (e.g. NSRR datasets).
        The XML file is assumed to have a list of events at a certain depth level.
        Each item in target_values will be treated separately so that the EventFrame will have an EventRecord for each key.
        If only target values are passed as an argument, the EventRecord will contain observed target values as states.

        Args:
            record_filepath (str): Path of the file to be parsed
            event_key (str): Which XML tag is associated to the type of event
            target_values (dict): Mapping of the target values of interest
            ts_key (str): Which XML tag is associated to the timing of the event
            t0 (datetime): Initial timestamp
            start_value (Any, optional): Initial value of the record. Defaults to None.
            value_function (Callable, optional): Set values according to elements in the data. Defaults to None (values are explicit from event_key)
            duration_key (str, optional): Which XML tag is associated to the timing of the event. Defaults to None -> spikes for binary events, automatically embedded for other events (e.g. sleep stages).
            events_path (list, optional): How to reach the right depth in the XML. Defaults to [] -> root of the document.
            ts_is_datetime (bool, optional): Flag if the ts column is absolute timestamps, or an increasing value from t0. Defaults to True.
            ts_sampling (float, optional): Sampling interval if timestamps are relative. Defaults to 1.0 seconds.

        Returns:
            [EventFrame]: EventFrame instance
        """
        start_date = t0
        output = cls(start_date=start_date)
        for key, val in target_values.items():
            output[key] = EventRecord.from_xml(record_filepath, key, event_key, val, ts_key, t0, start_value, value_function, duration_key, events_path, ts_is_datetime, ts_sampling)
        return output

    @classmethod
    def from_wfdb_annotation(cls, record_filepath:str, target_values:dict, t0:datetime, start_value:Any=None, extension:str='ann', openclose:Tuple=None):
        """Instantiate an EventFrame from a Physionet WFDB annotation file.
        Each item in target_values will be treated separately so that the EventFrame will have an EventRecord for each key.
        If only target values are passed as an argument, the EventRecord will contain observed target values as states.
        If openclose is passed, a binary state will be ON if the annotation starts with the opening character, OFF if it ends with closing one. e.g. '(apnea'->1, 'apnea)'->0

        Args:
            record_filepath (str): Path to the WFDB record folder with record name repeated e.g. sample_data/tr100/tr100
            label (str): Label assigned to the output EventRecord
            target_values (dict): Key of the EventFrame elements, with a set of values to be stored e.g. ['W', 'N1', 'N2'] or a single string for openclose scenarios e.g. 'apnea'
            t0 (datetime): Initial timestamp
            start_value (Any, optional): Override start value of the EventRecord. Defaults to None.
            extension (str, optional): Extension of the WFDB annotation file. Defaults to 'ann'.
            openclose (Tuple, optional): Tuple with opening and closing characters for events detection. Defaults to None.

        Returns:
            [EventFrame]: EventFrame instance
        """
        start_date = t0
        output = cls(start_date = start_date)
        for key,val in target_values.items():
            output[key] = EventRecord.from_wfdb_annotation(record_filepath, key, val, t0, start_value, extension, openclose)
        return output

    def merged_data(self, label:str='merged events', labels:List[str]=None, as_signal:bool=False, sampling_period:float=1.0) -> Union[EventRecord, Signal]:
        """Merge the events together. TODO allow external function to be used instead of logical or

        Args:
            label (str, optional): Output label. Defaults to 'merged events'.
            labels(List[str], optional): List of signals to be merged. Default to all signals if None
            as_signal (bool, optional): Return data as Signal instead of EventRecord. Defaults to False.
            sampling_period (float, optional): Desired sampling period if data is returned as Signal. Defaults to 1.0.

        Returns:
            Union[EventRecord, Signal]: Merged data
        """
        # Merge EventRecord timeseries
        if labels is not None:
            for val in filter(lambda x: x not in self.labels, labels):
                warnings.warn(f"Label {val} is not present in the EventFrame.")
        series = [x.data for x in self.values() if ((labels is None) or (x.label in labels))]
        all_binary_inputs = all([x.is_binary for x in self.values() if ((labels is None) or (x.label in labels))])
        temp_series = TimeSeries.merge(series, operation= lambda x: int(any(x))) if any([x.n_events>0 for x in self.values()]) else TimeSeries()

        # Transform to EventRecord
        start_value = temp_series.first_value() if not temp_series.is_empty() else 0
        start_value = start_value if start_value is not None else 0
        if temp_series.is_empty():
            temp_series[self.start_date] = start_value
        record = EventRecord(label=label, start_time=self.start_date, data=temp_series, start_value=start_value)
        record.is_binary = ((record.data.n_measurements()>1 and len(record.data.distribution().keys())<=2)) or all_binary_inputs

        if as_signal:
            return record.as_array(sampling_period)
        else:
            return record

    @property
    def n_events(self) -> int:
        """Return the total number of events in the EventFrame

        Returns:
            int: Total number of events
        """
        output = 0
        for x in self.values():
            output += x.n_events

        return output

def autocache(func:Callable, cache_folder:str, filename:str, cache_format:str='pickle', force:bool=False, ignore_kwargs:List[str]=None) -> Any:
    """Minimal wrapper to automatically manage cache of functions. Currently support pickle and matlab formats.
       BEWARE! Not all data formats may be supported correctly.
       Example call --> autocache(foo, 'test', 'test')(args)
       Cached call is tested against existence of cache file and hashing of source + passed arguments (not cached)
       Ignore_kwargs can be used to ignore certain keyword arguments that are passed to the argument. 
       An example is argument that may change at every call, but are not affecting the output result

    Args:
        func (Callable): function to be called.
        cache_folder (str): path to the cache folder.
        filename (str): filename of the cached data.
        cache_format (str, optional): Select which serialization format is used. Currently support 'pickle' and 'matlab'. Defaults to 'pickle'.
        force (bool, optional): Force the function to be executed even if cache exists. Defaults to False.
        ignore_kwargs (List[str], optional): Ignore certain keyword arguments. Defaults to None
        (args) Arguments of the function to be called

    Returns:
        Any: return of function 'func' with the same format
    """
    # File extensions
    file_extension = {
        'pickle': 'pkl',
        'matlab': 'mat'
    }

    def func_ref() -> str:
        """Get function name with package and version

        Returns:
            str: package(version).func_name string
        """
        try:
            module = inspect.getmodule(func)
            module_name = module.__name__
        except TypeError:
            module = func.__module__
            module_name = func.__module__
        try:
            module_version = version(module.__package__.split('.')[0])
        except (PackageNotFoundError, AttributeError):
            module_version = 'v0.0.0'
        try:
            func_name = func.__qualname__
        except AttributeError:
            func_name = func.__class__.__name__
        return f"{module_name}({module_version}).{func_name}"
    
    # Code hasher
    def hash(source:str, vars_data:dict, args_data:Any, kwargs_data:Any, ignore_kwargs:List[str]=None) -> str:
        """Hash function and arguments to check if they changed from cached version

        Args:
            source (str): result of inspect.getsource
            vars_data (dict): variables of the class (e.g. __init__ variables not passed to caller)
            args_data (Any): positional arguments
            kwargs_data (Any): keyword arguments
            ignore_kwargs (List[str], optional): Optional list of keyword arguments to be ignored. See docs above.

        Returns:
            str: hash result
        """
        hasher = blake2b(digest_size=32)
        hasher.update(source.encode('utf-8'))
        hasher.update(str([k+v.__repr__() for k,v in vars_data.items()]).encode('utf-8'))
        hasher.update(str([x.__repr__() for x in args_data]).encode('utf-8'))
        if ignore_kwargs is None:
            hasher.update(str(kwargs_data).encode('utf-8'))
        else:
            hasher.update(str({k:v for k,v in kwargs_data.items() if k not in ignore_kwargs}).encode('utf-8'))
        return hasher.hexdigest()

    # Function wrapper
    def wrap(*args, **kwargs):
        # Assert cache format
        assert cache_format in ['pickle', 'matlab'], "Invalid cache format. Valid options are 'pickle' or 'matlab'"

        # Get function reference and hash for later interpretability
        func_reference = func_ref()
        try:
            source = inspect.getsource(func)
        except TypeError:
            cls = getattr(importlib.import_module(func.__module__), func.__class__.__name__)
            source = inspect.getsource(cls)
        vars_data = func.__dict__ if hasattr(func, '__dict__') else {}
        hashed_call = hash(source, vars_data, args, kwargs, ignore_kwargs) 
        # Check if folder exists or create it
        if not Path(cache_folder).exists():
            os.makedirs(cache_folder, exist_ok=True)

        # Read data if not forced to call the function
        filepath = f"{cache_folder}/{filename}.{file_extension[cache_format]}"
        if (not force) and Path(filepath).exists():
            if cache_format == 'pickle':
                with open(filepath, "rb") as openfile:
                    loaded_data = pickle.load(openfile)
            elif cache_format == 'matlab':
                loaded_data = read_mat(filepath)
            data = loaded_data['data']
            loaded_hash = loaded_data['hash']

            # Return data if it was produced by the same function (source+arguments)
            if loaded_hash == hashed_call:
                return data

        # Call function
        output = func(*args, **kwargs)

        # Save data in cache
        save_obj = {'data': output, 'ref':func_reference, 'hash':hashed_call}
        if cache_format == 'pickle':
            with open(filepath, "wb") as openfile:
                pickle.dump(save_obj, openfile, protocol=pickle.HIGHEST_PROTOCOL)
        elif cache_format == 'matlab':
            savemat(filepath, save_obj)
        return output

    return wrap
