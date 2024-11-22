import evaluate
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from kg_mllm.test import evaluate_model
from kg_mllm.util.data import load_train_val_test

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


def create_model():
    config = AutoConfig.from_pretrained(model_name)
    model = BertForSequenceClassification(config=config)
    model.config.hidden_dropout_prob = 0.5
    print(model.adapter_summary())
    return model


def main() -> None:
    model = create_model()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset, test_dataset = load_train_val_test(tokenizer)

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

    trainer = Trainer(
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
