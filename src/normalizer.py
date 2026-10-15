"""
Persian Text Normalization and Preprocessing for ASR Evaluation and Training.
Handles:
- Character standardization (Arabic Kaf/Yeh -> Persian)
- Diacritics / Harakat removal
- Zero-Width Non-Joiner (ZWNJ / نیم‌فاصله) cleaning and optional space-conversion
- Number normalization
- Punctuation removal for standardized WER/CER
- Fallback pure-python normalizer if Hazm is unavailable
"""

import re
import unicodedata
from typing import Optional

try:
    from hazm import Normalizer as HazmNormalizer
    _HAZM_AVAILABLE = True
except ImportError:
    _HAZM_AVAILABLE = False


class PersianTextNormalizer:
    def __init__(self, use_hazm: bool = True):
        self.use_hazm = use_hazm and _HAZM_AVAILABLE
        if self.use_hazm:
            # Hazm API differs across versions/forks:
            # - sobhe/hazm 0.7-0.9: Normalizer() takes no args
            # - newer forks: Normalizer(remove_extra_spaces, persian_style, ...)
            # Try kwargs first, fall back to no-arg constructor.
            try:
                self.hazm = HazmNormalizer(
                    remove_extra_spaces=True,
                    persian_style=True,
                    persian_numbers=False,
                    remove_diacritics=True
                )
            except TypeError:
                try:
                    self.hazm = HazmNormalizer()
                except Exception:
                    self.hazm = None
                    self.use_hazm = False
            except Exception:
                self.hazm = None
                self.use_hazm = False
        else:
            self.hazm = None

        # Diacritics regex (harakat / tanwin)
        self.diacritics_pattern = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
        
        # Punctuation & symbols to remove for ASR evaluation
        self.punct_pattern = re.compile(r"""[.,?!:;"'()\[\]{}<>\/\\@#$%^&*_\-+=~`|«»؟،؛…—–]""")

        # Persian and Arabic digits mapping to standard Persian or ASCII
        # NOTE: "ۀ" (U+06C0 Heh with hamza / ezafe form) is intentionally NOT
        # mapped to "ه" — that would delete the ezafe. Hazm handles it.
        self.arabic_to_persian_chars = {
            "ك": "ک",
            "ي": "ی",
            "ى": "ی",
            "ة": "ه",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا",
            "آ": "آ",
            "\u0640": "",  # Tatweel / Kashida
        }

        # Colloquial markers only (formal words like چرا/چطور/اینجا/اونجا
        # also occur in formal text, so they are NOT included here).
        self.colloquial_patterns = [
            r"\bمی‌خوام\b", r"\bمیخوام\b", r"\bمی‌خواستم\b",
            r"\bمیرم\b", r"\bمیری\b", r"\bمیره\b", r"\bمیریم\b", r"\bمیرن\b",
            r"\bمیشه\b", r"\bنمیشه\b", r"\bمیشد\b",
            r"\bواسه\b", r"\bواسم\b", r"\bواست\b",
            r"\bچیکار\b",
            r"\bاونا\b", r"\bاینا\b",
            r"\bبگم\b", r"\bبگی\b", r"\bبگه\b",
            r"\bخونه\b", r"\bدیگه\b",
        ]
        self.colloquial_regex = re.compile("|".join(self.colloquial_patterns))

    def standardize_characters(self, text: str) -> str:
        """Replace Arabic glyphs with Persian counterparts and strip tatweel."""
        for ar, fa in self.arabic_to_persian_chars.items():
            text = text.replace(ar, fa)
        return text

    def remove_diacritics(self, text: str) -> str:
        """Remove short vowel marks (aerab / tanwin)."""
        return self.diacritics_pattern.sub("", text)

    def normalize_zwnj(self, text: str, mode: str = "keep") -> str:
        """
        Manage Zero-Width Non-Joiner (ZWNJ / \u200c).
        - mode='keep': clean duplicated or boundary ZWNJs
        - mode='to_space': replace ZWNJ with space (removes penalty for joined words in WER)
        - mode='remove': completely strip ZWNJ
        """
        if mode == "to_space":
            text = text.replace("\u200c", " ")
        elif mode == "remove":
            text = text.replace("\u200c", "")
        else:
            # Clean duplicate ZWNJs and spaces around ZWNJ
            text = re.sub(r"\u200c+", "\u200c", text)
            text = re.sub(r"\s+\u200c|\u200c\s+", " ", text)
            text = re.sub(r"^\u200c|\u200c$", "", text)
        return text

    def clean_whitespace(self, text: str) -> str:
        """Collapse multiple spaces and trim."""
        return re.sub(r"\s+", " ", text).strip()

    def normalize(
        self,
        text: str,
        remove_punctuation: bool = True,
        zwnj_mode: str = "keep"
    ) -> str:
        """
        Full text normalization pipeline for Persian ASR.
        
        Args:
            text: Raw input text.
            remove_punctuation: Whether to strip punctuation marks.
            zwnj_mode: One of 'keep', 'to_space', or 'remove'.
        """
        if not text:
            return ""

        # Step 1: Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Apply Hazm if available (guarded: never let Hazm crash training)
        if self.use_hazm and self.hazm is not None:
            try:
                text = self.hazm.normalize(text)
            except Exception:
                pass

        # Step 3: Standardize Arabic to Persian characters
        text = self.standardize_characters(text)

        # Step 4: Remove diacritics
        text = self.remove_diacritics(text)

        # Step 5: Remove punctuation if requested
        if remove_punctuation:
            text = self.punct_pattern.sub(" ", text)

        # Step 6: Process ZWNJ
        text = self.normalize_zwnj(text, mode=zwnj_mode)

        # Step 7: Collapse whitespace
        text = self.clean_whitespace(text)

        return text

    def is_colloquial(self, text: str) -> bool:
        """
        Heuristic rule-based check to identify if a Persian utterance contains
        colloquial/informal spoken markers.
        """
        if not text:
            return False
        return bool(self.colloquial_regex.search(text))
