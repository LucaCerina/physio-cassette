import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

import numpy as np
import pyedflib as edf
from pymatreader import read_mat

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
