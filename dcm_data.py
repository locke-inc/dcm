"""
dcm_data.py — Long-form text DataLoader for DCM
================================================
Serves continuous chunks of long-form text (e.g., Project Gutenberg books)
as (context, continuation) pairs for training the DCM pipeline.

Design choices:
    - Uses IterableDataset to stream text without loading entire corpora into RAM
    - Concatenates documents into a single token stream, then slices into
      fixed-length windows — zero padding waste
    - Each sample yields a context chunk and a continuation chunk from the
      *same* document stream, suitable for the hybrid training loop
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset


class LongTextIterableDataset(IterableDataset):
    """
    Streams tokenised long-form text as (context_ids, continuation_ids) pairs.

    Data flow:
        raw text files → tokenise → concatenate into buffer →
        slide window → yield (context, continuation) pairs

    The context window feeds the SSM encoder; the continuation window is
    what the AR head predicts (with labels shifted by 1 inside the model).

    Args:
        data_dir:        Directory containing .txt files
        tokenizer:       HuggingFace tokenizer instance
        context_len:     Number of tokens for the context (SSM encoder input)
        continuation_len:Number of tokens for the continuation (AR prediction)
        stride:          How far to advance the window each step (default = continuation_len)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        context_len: int = 1024,
        continuation_len: int = 512,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.continuation_len = continuation_len
        self.stride = stride or continuation_len
        self.total_len = context_len + continuation_len

    def _file_paths(self) -> list[str]:
        """Collect all .txt files in data_dir, sorted for reproducibility."""
        paths = []
        for root, _, files in os.walk(self.data_dir):
            for f in sorted(files):
                if f.endswith(".txt"):
                    paths.append(os.path.join(root, f))
        return paths

    def _tokenize_file(self, path: str) -> list[int]:
        """Read and tokenise a single text file."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __iter__(self) -> Iterator[dict]:
        """
        Yields dicts with:
            context_ids:        (context_len,)        LongTensor
            continuation_ids:   (continuation_len,)   LongTensor
            continuation_labels:(continuation_len,)   LongTensor (= continuation_ids shifted by 1)
        """
        worker_info = torch.utils.data.get_worker_info()
        file_paths = self._file_paths()

        # Shard files across DataLoader workers
        if worker_info is not None:
            per_worker = len(file_paths) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(file_paths)
            file_paths = file_paths[start:end]

        # Stream through files, maintaining a rolling token buffer
        buffer: list[int] = []

        for path in file_paths:
            tokens = self._tokenize_file(path)
            buffer.extend(tokens)

            # Yield windows while buffer is large enough
            while len(buffer) >= self.total_len + 1:  # +1 for the final label
                window = buffer[: self.total_len + 1]

                context_ids = torch.tensor(window[: self.context_len], dtype=torch.long)
                continuation_ids = torch.tensor(
                    window[self.context_len : self.total_len], dtype=torch.long
                )
                # Labels are the continuation shifted by 1 (next-token prediction)
                continuation_labels = torch.tensor(
                    window[self.context_len + 1 : self.total_len + 1], dtype=torch.long
                )

                yield {
                    "context_ids": context_ids,
                    "continuation_ids": continuation_ids,
                    "continuation_labels": continuation_labels,
                }

                buffer = buffer[self.stride :]

        # Handle final partial buffer if large enough
        if len(buffer) >= self.total_len + 1:
            window = buffer[: self.total_len + 1]
            context_ids = torch.tensor(window[: self.context_len], dtype=torch.long)
            continuation_ids = torch.tensor(
                window[self.context_len : self.total_len], dtype=torch.long
            )
            continuation_labels = torch.tensor(
                window[self.context_len + 1 : self.total_len + 1], dtype=torch.long
            )
            yield {
                "context_ids": context_ids,
                "continuation_ids": continuation_ids,
                "continuation_labels": continuation_labels,
            }


class SyntheticLongTextDataset(IterableDataset):
    """
    Generates synthetic random token sequences for testing / sanity checks.
    Useful when real data is not yet available.
    """

    def __init__(self, vocab_size: int, context_len: int, continuation_len: int, num_samples: int = 1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.continuation_len = continuation_len
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self.num_samples):
            total = self.context_len + self.continuation_len + 1
            tokens = torch.randint(0, self.vocab_size, (total,))
            yield {
                "context_ids": tokens[: self.context_len],
                "continuation_ids": tokens[self.context_len : self.context_len + self.continuation_len],
                "continuation_labels": tokens[self.context_len + 1 : total],
            }


def build_dataloader(
    data_dir: str,
    tokenizer,
    context_len: int = 1024,
    continuation_len: int = 512,
    batch_size: int = 2,
    num_workers: int = 2,
    stride: Optional[int] = None,
) -> DataLoader:
    """
    Convenience factory for creating the DCM training DataLoader.

    Uses pin_memory for faster host→GPU transfer on CUDA systems.
    """
    dataset = LongTextIterableDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        context_len=context_len,
        continuation_len=continuation_len,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
