import json
import os
from typing import Dict, List

import evaluate
import numpy as np
from adapters import AdapterTrainer, AutoAdapterModel, Trainer
from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

# TODO: Consolidate and move to config
language = 'FOO'
output_dir = './training_output'
adapter_dir = ''
model_name = 'bert-base-multilingual-cased'
learning_rate = 1e-4
num_train_epochs = 50
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
evaluation_strategy = 'epoch'
save_strategy = 'no'
weight_decay = 0.01


def encode_batch(examples: Dict[str, List]) -> Dict[str, List]:
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded: Dict[str, List] = {'input_ids': [], 'attention_mask': [], 'labels': []}
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

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


def preprocess_dataset(dataset: Dataset) -> Dataset:
    dataset = dataset.map(encode_batch, batched=True)
    dataset.set_format(columns=['input_ids', 'attention_mask', 'labels'])
    return dataset


def calculate_f1_on_test_set(
    trainer: Trainer, test_dataset: Dataset
) -> Dict[str, float]:
    print('Calculating F1 score on the test set...')
    test_predictions = trainer.predict(test_dataset)

    f1_metric = evaluate.load('f1')
    test_metrics = {
        'f1': f1_metric.compute(
            predictions=np.argmax(test_predictions.predictions, axis=-1),
            references=test_predictions.label_ids,
            average='macro',
        )['f1'],
    }

    print('Test F1 score:', test_metrics['f1'])
    return test_metrics


def main() -> None:
    dataset = load_dataset(f'dgurgurov/{language}_sa')
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    train_dataset = preprocess_dataset(train_dataset)
    val_dataset = preprocess_dataset(val_dataset)
    test_dataset = preprocess_dataset(test_dataset)

    # prepare model
    config = AutoConfig.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)

    # add task adapter
    model.add_adapter('sa')

    # set up task adapter
    model.add_classification_head('sa', num_labels=2)
    model.config.prediction_heads['sa']['dropout_prob'] = 0.5
    model.train_adapter(['sa'])
    print(model.adapter_summary())

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        weight_decay=weight_decay,
        output_dir=output_dir,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    f1_metric = evaluate.load('f1')

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: {
            'f1': f1_metric.compute(
                predictions=np.argmax(pred.predictions, axis=-1),
                references=pred.label_ids,
                average='macro',
            )['f1'],
        },
    )

    # train model
    trainer.train()

    # test model
    output_file_path = os.path.join(output_dir, 'test_metrics.json')
    with open(output_file_path, 'w') as json_file:
        json.dump(calculate_f1_on_test_set(trainer, test_dataset), json_file, indent=2)


if __name__ == '__main__':
    main()
