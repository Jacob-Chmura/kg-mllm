import json
from typing import Dict, Optional

import evaluate
import numpy as np
import Pathlib
from datasets import Dataset
from transfomers import Trainer


def evaluate_model(
    trainer: Trainer, test_dataset: Dataset, output_dir: Optional[Pathlib.Path]
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
    if output_dir is not None:
        output_file_path = output_dir / 'test_metrics.json'
        with open(output_file_path, 'w') as json_file:
            json.dump(test_metrics['f1'], json_file, indent=2)
    return test_metrics
