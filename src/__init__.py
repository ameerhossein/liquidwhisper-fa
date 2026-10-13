"""
Persian ASR Training & Evaluation Package
"""

from src.normalizer import PersianTextNormalizer
from src.metrics import ASRMetricsCalculator

try:
    from src.collator import DataCollatorSpeechSeq2SeqWithPadding
    from src.data import PersianASRDataPipeline
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    DataCollatorSpeechSeq2SeqWithPadding = None
    PersianASRDataPipeline = None

__all__ = [
    "PersianTextNormalizer",
    "ASRMetricsCalculator",
    "DataCollatorSpeechSeq2SeqWithPadding",
    "PersianASRDataPipeline",
]
