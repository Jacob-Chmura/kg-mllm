import evaluate
import numpy as np
from adapters import AdapterTrainer, AutoAdapterModel
from datasets import Dataset
from transfomers import Trainer
from transformers import TrainingArguments

# TODO: Consolidate and move to config
output_dir = './training_output'
learning_rate = 1e-4
num_train_epochs = 50
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
evaluation_strategy = 'epoch'
save_strategy = 'no'
weight_decay = 0.01


def train_model(
    model: AutoAdapterModel, train_dataset: Dataset, val_dataset: Dataset
) -> Trainer:
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
    return trainer
