import evaluate
import numpy as np
from adapters import AdapterConfig, AdapterTrainer, AutoAdapterModel
from adapters.composition import Stack
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

from kg_mllm.test import evaluate_model
from kg_mllm.util.data import tokenize_dataset

# TODO: Consolidate and move to config
language = 'FOO'
output_dir = './training_output'
adapter_dir = ''
adapter_config = 'BAR'
model_name = 'bert-base-multilingual-cased'
learning_rate = 1e-4
num_train_epochs = 50
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
evaluation_strategy = 'epoch'
save_strategy = 'no'
weight_decay = 0.01


def create_model():
    config = AutoConfig.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)

    # load pre-trained language adapter
    lang_adapter_config = AdapterConfig.load(adapter_config)
    model.load_adapter(
        adapter_dir,
        config=lang_adapter_config,
        load_as='lang_adapter',
        with_head=False,
    )

    # load down-stream task adapter
    model.add_adapter('sa')
    model.add_classification_head('sa', num_labels=2)

    # specify which adapter to train
    model.config.prediction_heads['sa']['dropout_prob'] = 0.5
    model.train_adapter(['sa'])

    # unfreeze and activate stack setup
    model.active_adapters = Stack('lang_adapter', 'sa')

    print(model.adapter_summary())
    return model


def main() -> None:
    model = create_model()

    dataset = load_dataset(f'dgurgurov/{language}_sa')
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        weight_decay=weight_decay,
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_total_limit=1,
        load_best_model_at_end=True,
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

    trainer.train()
    evaluate_model(trainer, test_dataset, output_dir)


if __name__ == '__main__':
    main()
