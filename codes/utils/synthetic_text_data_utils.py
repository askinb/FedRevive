#!/usr/bin/env python3
"""
Synthetic news generation + efficient labeled sampling utilities.

Example generation:
python synthetic_text_data_utils.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --sample_per_category 2000 \
    --max_new_tokens 128 \
    --temperature 2.0 \
    --top_k 100 \
    --top_p 0.95 \
    --seed 42 \
    --output ./../data/synthetic_news/synthetic_news.jsonl
"""

import os
import json
import random
import argparse
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import pipeline, set_seed, AutoTokenizer


CATEGORIES = {
    0: "atheism",
    1: "computer graphics",
    2: "windows os",
    3: "pc hardware",
    4: "mac hardware",
    5: "windows software",
    6: "for sale",
    7: "autos",
    8: "motorcycles",
    9: "baseball",
    10: "hockey",
    11: "cryptography",
    12: "electronics",
    13: "medicine",
    14: "space",
    15: "christianity",
    16: "gun politics",
    17: "middle east politics",
    18: "general politics",
    19: "religion",
}

SYSTEM_PROMPT = "You are a news writer."
USER_PROMPT_1 = "Generate a single news article about "
USER_PROMPT_2 = " without a title around "
USER_PROMPT_3 = " words."

# --------------------------------------------------------------------------- #
#                            TEXT GENERATION                                  #
# --------------------------------------------------------------------------- #

def build_prompt(topic: str, max_new_tokens: int) -> List[Dict]:
    """Construct chat-style prompt for TinyLlama."""
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {
            "role": "user",
            "content": (
                f"{USER_PROMPT_1}{topic}{USER_PROMPT_2}{int(max_new_tokens/2)}{USER_PROMPT_3}"
            ),
        },
    ]


def generate_synthetic_dataset(args):
    """Generate synthetic news data and save as JSONL."""
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading model {args.model} ...")
    pipe = pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        return_full_text=False,
    )

    with open(args.output, "w", encoding="utf-8") as fout:
        for label, topic in CATEGORIES.items():
            print(f"Generating samples for category {label}: {topic}")
            batch_prompts = []
            for i in range(args.sample_per_category):
                messages = build_prompt(topic, args.max_new_tokens)
                prompt = pipe.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_prompts.append(prompt)

            # Generate in batch
            outputs = pipe(
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batch_size=min(256, len(batch_prompts)),  # adjust for VRAM
            )

            for i, out in enumerate(outputs):
                text = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
                record = {
                    "label": label,
                    "topic": topic,
                    "text": text,
                    "sample_id": f"{label}_{i}",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")



    print(f"\nSaved synthetic dataset to {args.output}")


# --------------------------------------------------------------------------- #
#                            DATASET + SAMPLING                               #
# --------------------------------------------------------------------------- #

class SyntheticNewsDataset(Dataset):
    """Torch dataset for synthetic news data with efficient label sampling."""

    def __init__(
        self,
        path: str,
        tokenizer_name: str = "t5-small",
        max_length: int = 256,
        tokenize_on_load: bool = True,
    ):
        """
        Args:
            path: Path to JSONL file or directory containing per-label files.
            tokenizer_name: HuggingFace tokenizer to use.
            max_length: Tokenization max length.
            tokenize_on_load: Pre-tokenize all samples.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # --- Load JSONL(s)
        self.samples = []
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl")] \
            if os.path.isdir(path) else [path]
        for f in files:
            with open(f, "r", encoding="utf-8") as fin:
                for line in fin:
                    self.samples.append(json.loads(line))

        # --- Pre-tokenize if desired
        if tokenize_on_load:
            for ex in self.samples:
                enc = self.tokenizer(
                    ex["text"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                ex["input_ids"] = enc["input_ids"].squeeze(0)
                ex["attention_mask"] = enc["attention_mask"].squeeze(0)

        # --- Build label index for fast sampling
        self.by_label = {}
        for i, ex in enumerate(self.samples):
            lbl = ex["label"]
            self.by_label.setdefault(lbl, []).append(i)

        self.current_label = None  # optional focus label
        self.label_keys = sorted(list(self.by_label.keys()))

    def set_label(self, label: Optional[int]):
        """Set a label to sample exclusively from (None = all labels)."""
        self.current_label = label

    def __len__(self):
        if self.current_label is None:
            return len(self.samples)
        return len(self.by_label[self.current_label])

    def __getitem__(self, idx):
        if self.current_label is not None:
            idx = self.by_label[self.current_label][idx]
        ex = self.samples[idx]
        label = torch.tensor(ex["label"], dtype=torch.long)
        input_ids = ex.get("input_ids")
        if input_ids is None:
            enc = self.tokenizer(
                ex["text"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = enc["input_ids"].squeeze(0)
            ex["attention_mask"] = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": ex["attention_mask"],
            "label": label,
        }


class LabelSampler(Sampler):
    """Sampler that yields random indices from a specific label bucket."""

    def __init__(self, dataset: SyntheticNewsDataset, label: Optional[int] = None, shuffle=True):
        self.ds = dataset
        self.label = label
        self.shuffle = shuffle
        self.indices = (
            list(range(len(dataset)))
            if label is None
            else dataset.by_label[label]
        )

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloader(
    path: Optional[str] = None,
    ds: Optional[SyntheticNewsDataset] = None,
    batch_size: int = 8,
    label: Optional[int] = None,
    shuffle: bool = True,
    tokenizer_name: str = "t5-small",
    max_length: int = 256,
    tokenize_on_load: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Return DataLoader; if label given, sample only from that label."""
    if ds is None:
        ds = SyntheticNewsDataset(
        path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        tokenize_on_load=tokenize_on_load,
    )
    assert path is not None or ds is not None, "Either path or ds must be provided"
    sampler = LabelSampler(ds, label=label, shuffle=shuffle)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


# --------------------------------------------------------------------------- #
#                                 CLI Entrypoint                              #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sample_per_category", type=int, default=2000)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = "./../data/synthetic_news/synthetic_news.jsonl"

    generate_synthetic_dataset(args)


if __name__ == "__main__":
    main()
