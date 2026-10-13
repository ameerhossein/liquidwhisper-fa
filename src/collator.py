"""
Data collator for Speech-to-Text Sequence-to-Sequence models (Whisper / BuzzASR).
Handles dynamic padding of audio input features and target text tokens,
replacing padding tokens with -100 so CrossEntropyLoss ignores them.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    forward_attention_mask: bool = False

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and padding methods
        # Audio input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.tensor(
                [feature["attention_mask"] for feature in features],
                dtype=torch.long
            )

        # Target label ids
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is prepended in both tokenizer and model forward, trim if needed
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
