from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/forest_cover.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
def train(dataset_path: Path, random_state: int) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(labels=['Id','Cover_Type'],axis=1)
    target = dataset['Cover_Type']
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=train_test_split, random_state=random_state
    )
    pipeline = create_pipeline(random_state)
    pipeline.fit(features_train, target_train)
    accuracy = accuracy_score(target_val, pipeline.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
