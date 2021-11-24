import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

import numpy as np
import pyedflib as edf
from pymatreader import read_mat
from traces import TimeSeries

# A list of signals not to be loaded on TU/e laptops
***REMOVED***_DATA_SUBSET = ['TBD'] #TODO fill this

@dataclass
class Signal:
    """A class that holds any signal recorded from a sensor

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
    """
    label: str = ''
    data: np.ndarray = None
    fs: float = np.nan
    start_time: datetime = datetime.fromtimestamp(0)
    tstamps: np.ndarray = None

    def __getitem__(self, indexes):
        """Return a slice of self as a Signal object

        Args:
            indexes (slice): example sig[1:1000]

        Returns:
            [Signal]: sliced signal
        """
        return Signal(
            label=self.label,
            data=self.data[indexes],
            fs=self.fs,
            start_time=self.start_time, 
            tstamps=self.tstamps[indexes] if self.tstamps is not None else self.tstamps)

    def from_mat_file(self, mat_filename:str) -> Tuple[Any, str]:
        """Load a Signal from  NOPEformatted matlab file

        Args:
            mat_filename (str): filename

        Returns:
            Tuple[Any, str]: the loaded `py:class:~Signal` (None if it cannot be loaded), label of the signal

        Raises:
            ValueError if Start datetime variables are missing in the matlab file
        """
        label = Path(mat_filename).stem
        # TODO add assert for non mat files
        if os.environ['userdomain'] != 'CODE1' and label in ***REMOVED***_DATA_SUBSET: # Gatekeep for ***REMOVED*** data
            return None, label
        raw_mat = read_mat(mat_filename, variable_names=['data', 'SampleRate', 'StartDate', 'StartTime'])
        assert 'StartDate' in raw_mat.keys(), "ValueError: StartDate missing in Mat data file"
        assert 'StartTime' in raw_mat.keys(), "ValueError: StartTime missing in Mat data file"
        start_time = datetime.strptime(f"{raw_mat['StartDate']}-{raw_mat['StartTime']}", "%d/%m/%Y-%H:%M:%S")
        return Signal(label=label, data=raw_mat['data'], fs=raw_mat['SampleRate'], start_time=start_time), label

    @property
    def time(self):
        """Return time samples, calculate and store them if it wasn't done before

        Returns:
            [np.ndarray]: time instants of each sample
        """
        if self.tstamps is None:
            if not np.isnan(self.fs):
                steps = np.linspace(0, self.data.shape[0]/self.fs, self.data.shape[0])
                tsteps = [self.start_time + timedelta(seconds=x) for x in steps]
            else: # TODO strong assumption that it's a signal determined by events arrival (e.g. RR intervals)
                tsteps = [self.start_time + timedelta(seconds=x) for x in np.cumsum(self.data)]
            self.tstamps = np.array(tsteps)

        return self.tstamps

    @property
    def shape(self):
        return self.data.shape
    

class SignalFrame(dict):
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

    def from_mat_folder(self, folder:str, signal_names:Union[str,list]=None):
        """Generate a SignalFrame from a folder of matlab files, according to  NOPEformat

        Args:
            folder (str): name of the folder
            signal_names (Union[str,list], optional): restrict loading to certain signals only, otherwise load everything. Defaults to None.

        Returns:
            [self]: initialized SignalFrame
        """
        self.__init__()
        filenames = glob(folder+'/*.mat')
        for filename in filenames:
            signal, label = Signal().from_mat_file(mat_filename=filename)
            if signal and ((signal_names is None) or (label in signal_names)):
                self[label] = signal
                if self.start_date is None: # TODO assumption that all signals in the folder will have the same start time
                    self.start_date = signal.start_time
        return self

    # Labels property
    @property
    def labels(self):
        """Return label names

        Returns:
            [list]: label of all Signals in the SignalFrame
        """
        return [*self]

    def is_data_present(self, input_labels:Union[str, Iterable]) -> bool:
        """Check if every label given as input is present in the signalFrame

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
            
class DataHolder(dict):
    def __init__(self,*arg,**kw):
        super(DataHolder, self).__init__(*arg, **kw)
    
    # Labels property
    @property
    def labels(self):
        """Return label names

        Returns:
            [list]: label of all the data in the DataHolder
        """
        return [*self]

    def is_data_present(self, input_labels:Union[str, Iterable]) -> bool:
        """Check if every label given as input is present in the signalFrame

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
            flag variable to assess if the TimeSeries has only two possible states
        start_value: Any
            starting value at start_time
    """
    label: str = ''
    start_time: datetime = datetime.fromtimestamp(0)
    data: TimeSeries = TimeSeries()
    is_binary: bool = False
    start_value: Any = None

    def from_ts_dur_array(self, label:str, t0:datetime, ts_array:Union[list, np.ndarray], duration_array:Union[list, np.ndarray], is_binary:bool=False, start_value:Any=None):
        """Generate a EventRecord from two arrays, one with timestamps and one with duration of events. EventRecord may have bianry values and a start_value

        Args:
            label (str): label of the data instance
            t0 (datetime): initial datapoint, set to 0 if start_value is None
            ts_array (Union[list, np.ndarray]): timestamp array, can be datetimes or relative to t0
            duration_array (Union[list, np.ndarray]): duration array, assumed in seconds, should have the same length of ts_array
            is_binary (bool, optional): state if the Events have only two possible values. Defaults to False.
            start_value (Any, optional): Initial value at t0. Defaults to None.

        Returns:
            [self]: initialized EventFrame
        """
        # Transform lists to arrays
        _ts_array = np.asarray(ts_array)
        _duration_array = np.asarray(duration_array)
        assert _ts_array.shape[0] == _duration_array.shape[0], f"Timestamps and Duration arrays should have the same size, got {_ts_array.shape} and {_duration_array.shape}"

        # Fill values
        data = TimeSeries()
        data[t0] = 0 if start_value is None else start_value

        # Check for empty arrays
        if _ts_array.size == 0:
            return EventRecord(label=label, start_time=t0, data=data, is_binary=is_binary, start_value=start_value)

        # Timestamps are relative to start_time, assuming to be seconds
        if not isinstance(_ts_array[0], (datetime, np.datetime64)):
            for ts, dur in zip(_ts_array, _duration_array):
                t_start = self.start_time + timedelta(seconds=float(ts))
                t_end = t_start + timedelta(seconds=dur)
                data[t_start] = 1
                data[t_end] = 0
        else:
            for ts, dur in zip(_ts_array, _duration_array):
                t_start = ts
                t_end = t_start + timedelta(seconds=dur)
                data[t_start] = 1
                data[t_end] = 0

        return EventRecord(label=label, start_time=t0, data=data, is_binary=is_binary, start_value=start_value)

    def as_array(self, sampling_period:float) -> Signal:
        """Return data as a regularly sampled array

        Args:
            sampling_period (float): [description]

        Returns:
            np.ndarray: [description]
        """
        values = [y for _,y in self.data.sample(sampling_period=sampling_period)]
        return Signal(label = self.label, data = np.array(values), fs = 1/sampling_period, start_time=self.start_time)

class EventFrame(dict):
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
        super(EventFrame, self).__init__(*arg, **kw)

    def from_csv(self, filename:str, labels: Iterable, event_column: str, ts_column:str, duration_column:str, start_time:datetime=datetime.fromtimestamp(0), ts_is_datetime:bool=False, delimiter:str=','):
        """Instantiate an EventFrame from a CSV file, recording certain labels separately

        Args:
            filename (str): name of the file to be parsed
            labels (Iterable): desired labels to be extracted in the CSV file
            event_column (str): label of the event column
            ts_column (str): label of the timestamp column
            duration_column (str): label of the duration column
            start_time (datetime, optional): Initial datetime of the data. Defaults to datetime.fromtimestamp(0).
            ts_is_datetime (bool, optional): Flag to parse ts column as datetime and not t-t0. Defaults to False.
            delimiter (str, optional): CSV delimiter. Defaults to ','.

        Returns:
            [type]: [description]
        """
        self.start_date = start_time
        # Allocate temporary dictionary
        temp_dict = {}
        for label in labels:
            temp_dict[label] = {'ts':[], 'dur':[]}
        # Parse CSV file
        with open(filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=delimiter)
            for row in reader:
                if row[event_column] in labels:
                    key = row[event_column]
                    # Extract timestamps and duration
                    ts = datetime.fromisoformat(row[ts_column]) if ts_is_datetime else float(row[ts_column])
                    dur = float(row[duration_column])
                    temp_dict[key]['ts'].append(ts)
                    temp_dict[key]['dur'].append(dur)
        # Allocate the frames
        for key, data in temp_dict.items():
            self[key] = EventRecord().from_ts_dur_array(label=key, t0=self.start_date, ts_array=data['ts'], duration_array=data['dur'], is_binary=True)
        return self

    def merged_data(self, label:str='merged events', as_signal:bool=False, sampling_period:float=1.0) -> Union[EventRecord, Signal]:
        """Merge the events together. TODO allow external function to be used instead of logical or

        Args:
            label (str, optional): Output label. Defaults to 'merged events'.
            as_signal (bool, optional): Return data as Signal instead of EventRecord. Defaults to False.
            sampling_period (float, optional): Desired sampling period if data is returned as Signal. Defaults to 1.0.

        Returns:
            Union[EventRecord, Signal]: Merged data
        """
        # Merge EventRecord timeseries
        series = [x.data for x in self.values()]
        temp_series = TimeSeries.merge(series, operation= lambda x: int(any(x)))

        # Transform to EventRecord
        record = EventRecord(label=label, start_time=self.start_date, data=temp_series, start_value=temp_series.first_value())

        if as_signal:
            return record.as_array(sampling_period)
        else:
            return record

    # Labels property
    @property
    def labels(self):
        """Return label names

        Returns:
            [list]: label of all Signals in the EventFrame
        """
        return [*self]