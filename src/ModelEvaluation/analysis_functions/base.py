# @file base.py
# @author Maria Lesniewski 
# @desc Holds the Basic, Stateful and Configurable Function classes
from ..streaming_state import StreamingState
from utils.model_logger import Logger


class BasicAnalysis:
    """
    This is a class for regular functions 
    that can write final statistics / print to files
    while the data is still streaming. 
    Nothing special, but the running def lets us interface
    with them abstractly
    """
    
    def running(self, data_stream : StreamingState, recorder : Logger = None):
        """
        For math operations that can be done while 
        data is streaming. May or may not be recorded
        """
        raise NotImplementedError


class StatefulAnalysis(BasicAnalysis):
    """
    This is a class for analysis functions 
    that can't be done on the fly batchwise because they require
    global data aggregated statistics (e.g. the mean) before final analysis

    usually running method aggregates results into a class 'state' attribute
    final combines those states into the final calculation 

    Attributes:
        state - a place for holding aggregrated statistics 
    """

    def __init__(self):
        super().__init__()
        self.state = {}
    
    def final(self, recorder : Logger):
        """
        This is for finalizing statistics of the analysis from state
        (post streaming) 
        """
        raise NotImplementedError

class ConfigurableAnalysis(StatefulAnalysis):
    """
    This is a class for analysis functions 
    that require special metadata (a configuration) to be executed 
    in a stream + finalize fashion
    e.g. info about the min/max in a histogram can help create batchwise data
    and allow it to be globally aggregatable

    Attributes:
        -state (see StatefulAnalysis)
        -config - a place for holding metaparameters required for analysis
    """

    def __init__(self):
        super().__init__()

        self.config = {}

    def obtain_config(self):
        """
        Function for explicitly retrieving config
        for calculation from external data source
        """

        raise NotImplementedError
