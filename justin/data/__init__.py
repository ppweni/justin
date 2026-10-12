from .base import Sample, TimeSeriesDataset
from .ecg import ECGDataset
from .har import HARDataset
from .sleep import SleepDataset
from .tsqa import TSQADataset

DATASETS = {"har": HARDataset, "sleep": SleepDataset, "tsqa": TSQADataset, "ecg": ECGDataset}

__all__ = [
    "DATASETS",
    "ECGDataset",
    "HARDataset",
    "Sample",
    "SleepDataset",
    "TSQADataset",
    "TimeSeriesDataset",
]
