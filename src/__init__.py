"""
Persian ASR Training & Evaluation Package
"""

# Load .env (repo root / CWD) before anything reads os.environ. Stdlib-only,
# never overrides real environment variables.
try:
    from .dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

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
