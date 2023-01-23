from __future__ import annotations
# -*- coding: utf-8 -*-
__author__      = "Luca Cerina"
__copyright__   = "Copyright 2022, Luca Cerina"
__email__       = "lccerina@gmail.com"

"""
This module implements simple abstractions over data and I/O to store signals, time series, metadata and event annotations.
Designed for physiological signals, but general enough to be used with any type of data.
"""

import csv
import inspect
import pickle
import warnings
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from logging import warning
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
import pyedflib as edf
from pymatreader import read_mat
from scipy.io import savemat
from traces import TimeSeries
from operator import *


# Default matlab format for signals
MATLAB_DEFAULT_SIGNAL_FORMAT = {
    'data': 'data',
    'sampling_rate': 'SampleRate',
    'start_date': 'StartDate',
    'start_time': 'StartTime'
}

class DataHolder(dict):
    """Minor abstraction over dict class to hold information together and check labels
    """
    def __init__(self,*arg,**kw):
        super(DataHolder, self).__init__(*arg, **kw)

    def relabel(self, mapper:dict):
        """Update labels

        Args:
            mapper (dict): Dictionary with new labels
        """
        for key, val in mapper.items():
            if key in self:
                self[val] = self.pop(key)
                self[val].label = val
    
    def is_data_present(self, input_labels:Union[str, Iterable]) -> bool:
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
        return [*self]


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
    fs: float = np.nan
    start_time: datetime = datetime.fromtimestamp(0)
    tstamps: np.ndarray = None

    def __post_init__(self):
        self.data = np.array(self.data) if self.data is not None else np.array([])
        self.tstamps = np.array(self.tstamps) if self.tstamps is not None else np.array([])

    def __setitem__(self, indexes:slice, values:Union[np.ndarray, Number]):
        """Update values of a slice of the Signal

        Args:
            indexes (slice): example sig[1:1000]
            values (Union[np.ndarray, Number]): new values
        """
        assert isinstance(indexes, slice) or (isinstance(indexes, np.ndarray) and np.issubdtype(indexes[0], np.bool_))
        assert isinstance(values, np.ndarray) or isinstance(values, Number)
        self.data[indexes] = values

    def sub(self, indexes:Union[slice,Iterable]=None, stop:Any=None, start:Any=None, step:Any=None, reset_start_time:bool=True):
        """Return a slice of self as a Signal object

        Args:
            indexes (slice, Iterable): example slice(None,10,None), slice (1, 10, None), [1,10,2]. Single values are not accepted, use __getitem__ slicing

        Returns:
            [Signal]: sliced signal
        """
        assert (indexes is not None) ^ ((stop is not None) or (start is not None and stop is not None)), "At least indexes, stop, or start and stop should be assigned, but not together"
        assert ~np.issubdtype(type(indexes), np.integer), "Please refrain from calling sub on a single value, use your_signal[idx] instead"
        assert isinstance(indexes, slice) or (isinstance(indexes, Iterable) and len(indexes)<=3), "Invalid indexes, use slice object or a iterable with a maximum length of 3"

        # Define indexes
        if indexes is not None:
            _indexes = indexes if isinstance(indexes, slice) else slice(*indexes)
        else:
            _indexes = slice(start, stop, step)

        # Check for start time slicing
        if reset_start_time:
            _start_time = self.tstamps[_indexes.start] if self.tstamps is not None else self.start_time + timedelta(seconds=_indexes.start/self.fs)
        else:
            _start_time = self.start_time

        return Signal(
            label=self.label,
            data=self.data[_indexes],
            fs=self.fs,
            start_time=_start_time,
            tstamps=self.tstamps[_indexes] if self.tstamps is not None else self.tstamps
        )

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

    def __unary_op__(self:Signal, other:Union[int, float, np.ndarray, Signal], op:function) -> Signal:
        """Apply a unary operation, accounting for edge cases

        Args:
            self (Signal): Input Signal
            other (Union[int, float, np.ndarray, Signal]): other operand
            op (function): operation

        Returns:
            Signal: Result of the operation
        """
        output = Signal(**vars(self))
        if np.issubdtype(type(other), np.number):
            output.data = op(self.data, other)
        elif isinstance(other, np.ndarray):
            if len(self)==len(other) and self.ndim==other.ndim:
                output.data = op(self.data, other)
            else:
                raise ValueError(f"The Signal object and the array should have same size and dimensions, got {self.shape} and {other.shape}")
        elif isinstance(other, Signal):
            if self.fs==other.fs:
                overlap_start, self_slice, other_slice = self.__get_overlap__(other)
                output.data = []
                if overlap_start is not None:
                    output.data = op(self[self_slice],other[other_slice])
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
        return self.__unary_op__(other, mul)

    def __add__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, add)

    def __radd__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, add)

    def __sub__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, sub)

    def __rsub__(self:Signal, other:Union[int, float, np.ndarray, Signal]) -> Signal:
        return self.__unary_op__(other, sub)

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

    def from_mat_file(self, mat_filename:str, data_format:dict=MATLAB_DEFAULT_SIGNAL_FORMAT, time_format:str="%d/%m/%Y-%H:%M:%S") -> Tuple[Any, str]:
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
            warning(f"{e} {mat_filename}")
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

    def __init__(self,*arg,**kw):
        super(SignalFrame, self).__init__(*arg, **kw)

    def from_arrays(self, labels:Iterable, signals:Iterable, samplings:Iterable, start_time:datetime=datetime.fromtimestamp(0)):
        """Generate a SignalFrame from multiple arrays

        Args:
            labels (Iterable): label for each Signal
            signals (Iterable): data for each Signal
            samplings (Iterable): sampling frequency for each Signal
            start_time (datetime, optional): initial datetime. Defaults to datetime.fromtimestamp(0).

        Returns:
            [self]: initialized SignalFrame
        """
        self.__init__()
        self.start_date = start_time
        for label, signal, fs in zip(labels,signals,samplings):
            self[label] = Signal(label=label, data=signal, fs=fs)
        return self
    
    def from_edf_file(self, edf_filename:str, signal_names:Union[str,list]=None):
        """Generate a SignalFrame from a single EDF/BDF file

        Args:
            edf_filename (str): filename
            signal_names (Union[str,list], optional): restrict loading to certain signals only, otherwise load everything. Defaults to None.

        Returns:
            [self]: initialized SignalFrame
        """

        self.__init__()
        # Read edf file
        signals, signal_headers, header = edf.highlevel.read_edf(edf_filename)
        self.start_date = header['startdate']
        for sig_header, signal in zip(signal_headers, signals):
            label = sig_header['label']
            if (signal_names is None) or (label in signal_names):
                self[label] = Signal(label=label, data=signal, fs=sig_header['sample_rate'], start_time=self.start_date)
        return self

    def from_mat_folder(self, folder:str, signal_names:Union[str,list]=None, data_format:dict=MATLAB_DEFAULT_SIGNAL_FORMAT, time_format:str="%d/%m/%Y-%H:%M:%S"):
        """Generate a SignalFrame from a folder of matlab files, according to a certain format

        Args:
            folder (str): name of the folder
            signal_names (Union[str,list], optional): restrict loading to certain signals only, otherwise load everything. Defaults to everything.
            data_format (dict, optional): Name of the variables in the mat files. Defaults to MATLAB_DEFAULT_SIGNAL_FORMAT.
            time_format (str, optional): format used to parse date and time string. Defaults to "%d/%m/%Y-%H:%M:%S".

        Returns:
            [self]: initialized SignalFrame
        """
        self.__init__()
        filenames = glob(folder+'/*.mat')
        for filename in filenames:
            label = Path(filename).stem
            if (signal_names is None) or (label in signal_names):
                signal, label = Signal().from_mat_file(mat_filename=filename, data_format=data_format, time_format=time_format)
                if signal:
                    self[label] = signal
                    # Start date is assumed to be common for all signals, otherwise consider the oldest one
                    if (self.start_date is None) or (signal.start_time < self.start_date): 
                        self.start_date = signal.start_time
        return self

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
    data: TimeSeries = TimeSeries()
    is_binary: bool = False
    is_spikes: bool = False
    start_value: Any = None

    def __len__(self):
        """Override __len__ to return the number of events.
        In binary series first 'non-event' is subtracted. Start and end of event (state transitions) are considered together, so divided by 2
        In non binary series returns number of measurements.

        Returns:
            int: Number of events
        """
        return (self.data.n_measurements()-1)//2 if self.is_binary else self.data.n_measurements()

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
            [EventRecord]: _description_
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
        return EventRecord(label=self.label, start_time=tstart, data=self.data.slice(tstart, tend), is_binary=self.is_binary, start_value=self.start_value)

    def from_logical_array(self, label:str, t0:datetime, input_array:np.ndarray, fs:float=1):
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
            return self.from_ts_dur_array(label=label, t0=t0, ts_array=[], duration_array=[], is_binary=True, start_value=0)

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

        return self.from_ts_dur_array(label=label, t0=t0, ts_array=ts, duration_array=durations, is_binary=True, start_value=0)

    def from_state_array(self, label:str, t0:datetime, input_array:Iterable, ts_sampling:float=1.0):
        """Generate an EventRecord from an array of states, with a fixed sampling time in seconds.

        Args:
            label (str): label of the data instance
            t0 (datetime): initial timestamp
            input_array (Iterable): list of events
            ts_sampling (float, optional): Desired sampling time. Defaults to 1.0 seconds.

        Returns:
            [self]: initialized EventRecord
        """
        if isinstance(input_array, np.ndarray):
            assert input_array.ndim==1, "EventRecord from_state_array accepts only 1D arrays"

        if len(input_array)==0:
            warnings.warn("Empty input in EventRecord data")
            return self.from_ts_dur_array(label=label, t0=t0, ts_array=[], duration_array=[], is_binary=True, start_value=0)

        data = TimeSeries()
        for i, val in enumerate(input_array):
            ts = t0 + timedelta(seconds=i*ts_sampling)
            data[ts] = val
        data.compact()

        return EventRecord(label=label, start_time=t0, data=data, is_binary=False, start_value=data[t0])


    def from_ts_dur_array(self, label:str, t0:datetime, ts_array:Union[list, np.ndarray], duration_array:Union[list, np.ndarray]=None, is_binary:bool=False, is_spikes:bool=False, start_value:Any=None):
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
        data[t0] = 0 if start_value is None else start_value
        self.start_time = t0

        # Check for empty arrays
        if _ts_array.size == 0:
            return EventRecord(label=label, start_time=t0, data=data, is_binary=is_binary, start_value=start_value)

        # Timestamps are relative to start_time, assuming to be seconds
        rel_ts_flag = not isinstance(_ts_array[0], (datetime, np.datetime64))
        delta_dur_flag = isinstance(_duration_array[0], timedelta)
        for ts, dur in zip(_ts_array, _duration_array):
            t_start = self.start_time + timedelta(seconds=ts) if rel_ts_flag else ts
            data[t_start] = 1
            if isinstance(dur, timedelta) or ~np.isnan(dur):
                dur_seconds = dur.total_seconds() if delta_dur_flag else dur
                t_end = (self.start_time + timedelta(seconds=ts+dur_seconds)) if rel_ts_flag else (ts + timedelta(seconds=dur_seconds)) 
                data[t_end] = 0

        return EventRecord(label=label, start_time=t0, data=data, is_binary=is_binary, is_spikes=_is_spikes, start_value=start_value)

    def from_csv(self, filename:str, label:str, event_column:str, t0:datetime, start_value:Any=None, ts_column:str=None, ts_is_datetime:bool=True, ts_sampling:float=0,  delimiter:str=',', skiprows:int=0, **kwargs):
        r"""Instantiate an EventRecord from a CSV file

        Args:
            filename (str): name of the file to be parsed
            label (str): label assigned to the output EventRecord
            event_column (str): label of the event column in the CSV
            t0 (datetime): initial timestamp
            start_value (Any, optional): Initial value of the record. Defaults to None.
            ts_column (str, optional): label of the timestamps column in the CSV. Defaults to None.
            ts_is_datetime (bool, optional): Flag if the ts column is absolute timestamps, or an increasing value from t0. Defaults to True.
            ts_sampling (float, optional): sampling interval if timestamps are relative. Defaults to 0.
            delimiter (str, optional): CSV column delimiter. Defaults to ','.
            skiprows (int, optional): Skip a certain number of rows before the csv reader
            \**kwargs: keywords arguments passed to csv reader (e.g. skiprows)

        Returns:
            [EventRecord]: EventRecord instance
        """
        assert skiprows>=0, "Skiprows argument should be positive"

        # Fill values
        data = TimeSeries()      
        self.start_time = t0

        if ts_column is None and ts_sampling == 0:
            raise(ValueError("if ts_column is None a valid ts_sampling in seconds should be passed to the function"))

        with open(filename, 'r') as csv_file:
            for _ in range(skiprows):
                csv_file.readline()
            reader = csv.DictReader(csv_file, delimiter=delimiter)
            for row in reader:
                value = row[event_column]
                ts = datetime.fromisoformat(row[ts_column]) if ts_is_datetime else self.start_time + timedelta(seconds=(int(row[ts_column])-1)*ts_sampling)
                data[ts] = value
        return EventRecord(label=label, start_time=t0, data=data, is_binary=False, start_value=start_value)    

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
        n_samples = int(np.floor((self.data.last_key() - self.start_time).total_seconds()/sampling_period)) if self.data.last_key()>self.start_time else 1
        values = np.zeros((n_samples,))
        if self.is_spikes == False:
            # Regular data is resampled without binning
            sample_delta = timedelta(seconds=sampling_period)
            values[0] = self.start_value
            for i in range(1, n_samples):
                curr_t = self.start_time + timedelta(seconds=i*sampling_period)
                if curr_t in self.data._d:
                    values[i] = self.data[curr_t]
                else:
                    right_index = self.data._d.bisect_right(curr_t)
                    prev_item = self.data.get_item_by_index(right_index-1)
                    near_flag = (curr_t - prev_item[0]) <= sample_delta
                    values[i] = prev_item[1] if near_flag else values[i-1]
        else:
            # Assign to closest samples
            for t,_ in self.data.items():
                t_sample = np.clip(int(np.round((t-self.start_time).total_seconds()/sampling_period)), None, n_samples-1)
                values[t_sample] = 1

        output_dtype = np.bool8 if self.is_binary else type(self.data.first_value())
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


    def remap(self, map:Union[dict, Callable]) -> None:
        """Update values of events inplace with a dictionary mapping or a function

        Args:
            map (Union[dict, Callable]): Mapping dictionary or Callable/lambda

        """
        assert isinstance(map, dict) or isinstance(map, Callable), "Mapping should be a dict or a function"
        if isinstance(map, dict):
            for t, val in self.data:
                new_val = map[val]
                self.data[t] = new_val
        else:
            self.data = self.data.operation(None, lambda x,y: map(x))

    def binarize(self, key:str, compact:bool=True):
        """Remap helper, returns a binary EventRecord with only 1 where a key is present, 0 elsewhere

        Args:
            key (str): _description_
            compact (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
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

class EventFrame(DataHolder):
    """A class to hold various EventRecord together
    
    Attributes:
        start_date: datetime
            start datetime, considered equal for all EventRecords
        labels: list[str]
            list of labels for all Signals
        dict object: dict
            Signals are stored as a dict with label:Signal format
    """
    start_date = None

    def __init__(self,*arg,**kw):
        self.start_date = kw.pop('start_date', None)
        super(EventFrame, self).__init__(*arg, **kw)

    def from_csv(self, filename:str, labels: Iterable, event_column: str, ts_column:str, duration_column:str=None, start_time:datetime=datetime.fromtimestamp(0), ts_is_datetime:bool=False, delimiter:str=',', skiprows:int=0, **kwargs):
        r"""Instantiate an EventFrame from a CSV file, recording certain labels separately

        Args:
            filename (str): name of the file to be parsed
            labels (Iterable): desired labels to be extracted in the CSV file
            event_column (str): label of the event column
            ts_column (str): label of the timestamp column
            duration_column (str): label of the duration column. If None store as spike EventRecord
            start_time (datetime, optional): Initial datetime of the data. Defaults to datetime.fromtimestamp(0).
            ts_is_datetime (bool, optional): Flag to parse ts column as datetime and not t-t0. Defaults to False.
            delimiter (str, optional): CSV delimiter. Defaults to ','.
            skiprows (int, optional): Skip a certain number of rows before the csv reader
            \**kwargs: keywords arguments passed to csv reader (e.g. skiprows)

        Returns:
            [EventFrame]: EventFrame instance
        """
        assert skiprows>=0, "Skiprows argument should be positive"

        self.start_date = start_time
        # Allocate temporary dictionary
        temp_dict = {}
        for label in labels:
            temp_dict[label] = {'ts':[], 'dur':[]}
        # Parse CSV file
        with open(filename, 'r') as csv_file:
            for _ in range(skiprows):
                csv_file.readline()
            reader = csv.DictReader(csv_file, delimiter=delimiter)
            for row in reader:
                if row[event_column] in labels:
                    key = row[event_column]
                    # Extract timestamps and duration
                    ts = datetime.fromisoformat(row[ts_column]) if ts_is_datetime else float(row[ts_column])
                    temp_dict[key]['ts'].append(ts)
                    if duration_column is not None:
                        dur = float(row[duration_column])
                        temp_dict[key]['dur'].append(dur)
        # Allocate the frames
        for key, data in temp_dict.items():
            _is_spikes = len(data['dur']) == 0
            self[key] = EventRecord().from_ts_dur_array(label=key, t0=self.start_date, ts_array=data['ts'], duration_array=data['dur'], is_binary=True, is_spikes=_is_spikes)
        return self

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
            for val in labels:
                if val not in self.labels:
                    warnings.warn(f"Label {val} is not present in the EventFrame.")
        series = [x.data for x in self.values() if ((labels is None) or (x.label in labels))]
        temp_series = TimeSeries.merge(series, operation= lambda x: int(any(x)))

        # Transform to EventRecord
        start_value = temp_series.first_value() if not temp_series.is_empty() else 0
        record = EventRecord(label=label, start_time=self.start_date, data=temp_series, start_value=start_value)
        record.is_binary = True if record.data.n_measurements()>1 and len(record.data.distribution().keys())<=2 else False

        if as_signal:
            return record.as_array(sampling_period)
        else:
            return record

    @property
    def n_events(self) -> int:
        """Return the total number of respiratory events

        Returns:
            int: Total number of events
        """
        output = 0
        for x in self.values():
            output += x.n_events

        return output

def autocache(func:Callable, cache_folder:str, filename:str, cache_format:str='pickle', force:bool=False) -> Any:
    """Minimal wrapper to automatically manage cache of functions. Currently support pickle and matlab formats.
       BEWARE! Not all data formats may be supported correctly.
       Example call --> autocache(foo, 'test', 'test')(args)

    Args:
        func (Callable): function to be called.
        cache_folder (str): path to the cache folder.
        filename (str): filename of the cached data.
        cache_format (str, optional): Select which serialization format is used. Currently support 'pickle' and 'matlab'. Defaults to 'pickle'.
        force (bool, optional): Force the fucntion to be executed even if cache exists. Defaults to False.
        (args) Arguments of the function to be called

    Returns:
        Any: return of function 'func' with the same format
    """
    # File extensions
    file_extension = {
        'pickle': 'pkl',
        'matlab': 'mat'
    }
    # Function wrapper
    def wrap(*args, **kwargs):
        # Assert cache format
        assert cache_format in ['pickle', 'matlab'], "Invalid cache format. Valid options are 'pickle' or 'matlab'"

        # Get function doc for later interpretability
        func_doc = inspect.getdoc(func)

        # Check if folder exists or create it
        if not Path(cache_folder).exists():
            Path(cache_folder).mkdir()

        # Read data if not forced to call the function
        filepath = f"{cache_folder}/{filename}.{file_extension[cache_format]}"
        if (not force) and Path(filepath).exists():
            if cache_format == 'pickle':
                with open(filepath, "rb") as openfile:
                    data = pickle.load(openfile)['data']
            elif cache_format == 'matlab':
                data = read_mat(filepath)['data']

            # No transformation on data if the information is not available
            return data

        # Call function
        output = func(*args, **kwargs)

        # Save data in cache
        save_obj = {'data': output, 'doc':func_doc}
        if cache_format == 'pickle':
            with open(filepath, "wb") as openfile:
                pickle.dump(save_obj, openfile, protocol=pickle.HIGHEST_PROTOCOL)
        elif cache_format == 'matlab':
            savemat(filepath, save_obj)
        return output

    return wrap
