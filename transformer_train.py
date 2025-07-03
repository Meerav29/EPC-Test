"""Fine-tune DistilBERT for question classification.

This script mirrors the data split used in ``load.py`` but trains a transformer
model using Hugging Face ``transformers``. It expects ``pandas``, ``scikit-learn``,
``datasets`` and ``torch`` to be installed.
"""

import re
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def load_dataset(path: str) -> Dict[str, Dataset]:
    df = pd.read_csv(path)
    df["text"] = df["Sample Question"].apply(clean)

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Category"])

    train_df, test_df = train_test_split(
        df[["text", "label"]], test_size=0.25, random_state=42, stratify=df["label"]
    )
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    return {
        "train": train_ds,
        "test": test_ds,
        "labels": list(label_encoder.classes_),
    }


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding=True, truncation=True)


def train_transformer(data_path: str = "updated-questions-580.csv") -> None:
    data = load_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    encoded_train = data["train"].map(lambda x: tokenize(x, tokenizer), batched=True)
    encoded_test = data["test"].map(lambda x: tokenize(x, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(data["labels"])
    )

    training_args = TrainingArguments(
        output_dir="bert-model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_test,
        tokenizer=tokenizer,
    )

    trainer.train()
    preds = trainer.predict(encoded_test)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = encoded_test["label"]
    report = classification_report(y_true, y_pred, target_names=data["labels"])
    print("Transformer Classification Report:\n", report)


if __name__ == "__main__":
    train_transformer()
