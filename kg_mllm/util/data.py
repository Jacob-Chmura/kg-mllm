from typing import Dict, List

from datasets import Dataset
from transformers import Tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer: Tokenizer) -> Dataset:
    dataset = dataset.map(lambda sample: _encode_batch(sample, tokenizer), batched=True)
    dataset.set_format(columns=['input_ids', 'attention_mask', 'labels'])
    return dataset


def _encode_batch(examples: Dict[str, List], tokenizer: Tokenizer) -> Dict[str, List]:
    all_encoded: Dict[str, List] = {'input_ids': [], 'attention_mask': [], 'labels': []}

    for text, label in zip(examples['text'], examples['label']):
        encoded = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
        )
        all_encoded['input_ids'].append(encoded['input_ids'])
        all_encoded['attention_mask'].append(encoded['attention_mask'])
        all_encoded['labels'].append(label)

    return all_encoded
