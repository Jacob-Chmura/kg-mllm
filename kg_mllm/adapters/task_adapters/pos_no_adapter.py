from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForSequenceClassification,
)

from kg_mllm.test import evaluate_model
from kg_mllm.train import train_model
from kg_mllm.util.data import load_train_val_test

# TODO: Consolidate and move to config
language = 'FOO'
output_dir = './training_output'
adapter_dir = ''
model_name = 'bert-base-multilingual-cased'


def create_model():
    config = AutoConfig.from_pretrained(model_name)
    model = BertForSequenceClassification(config=config)
    model.config.hidden_dropout_prob = 0.5
    print(model.adapter_summary())
    return model


if __name__ == '__main__':
    model = create_model()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset, test_dataset = load_train_val_test(tokenizer)
    trainer = train_model(model, train_dataset, val_dataset)
    evaluate_model(trainer, test_dataset, output_dir)
