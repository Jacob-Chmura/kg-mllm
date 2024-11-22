from adapters import AdapterConfig, AutoAdapterModel
from adapters.composition import Stack
from transformers import AutoConfig, AutoTokenizer

from kg_mllm.test import evaluate_model
from kg_mllm.train import train_model
from kg_mllm.util.data import load_train_val_test

# TODO: Consolidate and move to config
output_dir = './training_output'
adapter_dir = ''
adapter_config = 'BAR'
model_name = 'bert-base-multilingual-cased'


def create_model() -> AutoAdapterModel:
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


if __name__ == '__main__':
    model = create_model()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset, test_dataset = load_train_val_test(tokenizer)
    trainer = train_model(model, train_dataset, val_dataset)
    evaluate_model(trainer, test_dataset, output_dir)
