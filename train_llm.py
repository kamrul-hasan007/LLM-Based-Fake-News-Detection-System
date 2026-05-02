import os
import re
import csv
import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fake_news_merged.csv"
SAVE_DIR = BASE_DIR / "saved_model"
SAVE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

# CPU-friendly settings
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5

# For testing on CPU, keep 5000.
# For better result, use 10000.
# For full dataset, use None, but it will be slow on CPU.
MAX_ROWS = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_column_name(col):
    col = str(col).strip()
    col = re.sub(r"[^A-Za-z0-9_]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def encode_label(value):
    value = str(value).strip().lower()

    fake_labels = {
        "0",
        "fake",
        "false",
        "f",
        "unreliable",
        "misinformation"
    }

    real_labels = {
        "1",
        "real",
        "true",
        "t",
        "reliable"
    }

    if value in fake_labels:
        return 0

    if value in real_labels:
        return 1

    try:
        numeric_value = int(float(value))

        if numeric_value == 0:
            return 0

        if numeric_value == 1:
            return 1

    except Exception:
        pass

    return np.nan


# ============================================================
# Robust CSV Loading Functions
# ============================================================

def try_pandas_read_csv():
    """
    Tries multiple safer pandas CSV reading methods.
    This handles many corrupted CSV rows automatically.
    """

    read_attempts = [
        {
            "usecols": ["title", "text", "label"],
            "engine": "python",
            "encoding": "utf-8",
            "encoding_errors": "replace",
            "on_bad_lines": "skip",
            "quotechar": '"',
            "doublequote": True,
            "escapechar": "\\"
        },
        {
            "usecols": ["title", "text", "label"],
            "engine": "python",
            "encoding": "latin1",
            "on_bad_lines": "skip",
            "quotechar": '"',
            "doublequote": True,
            "escapechar": "\\"
        },
        {
            "engine": "python",
            "encoding": "utf-8",
            "encoding_errors": "replace",
            "on_bad_lines": "skip",
            "quotechar": '"',
            "doublequote": True,
            "escapechar": "\\"
        },
        {
            "engine": "python",
            "encoding": "latin1",
            "on_bad_lines": "skip",
            "quotechar": '"',
            "doublequote": True,
            "escapechar": "\\"
        }
    ]

    last_error = None

    for attempt_number, attempt in enumerate(read_attempts, start=1):
        try:
            print(f"Trying pandas CSV read method {attempt_number}...")
            df = pd.read_csv(DATA_PATH, **attempt)
            print("Pandas CSV read successful.")
            return df

        except Exception as e:
            last_error = e
            print(f"Method {attempt_number} failed:", e)

    raise last_error


def manual_csv_repair_reader():
    """
    Fallback reader for heavily corrupted CSV files.

    Your dataset header appears like:
    Unnamed: 0,title,text,label,...

    This function reads the file line by line, detects each new row by the
    first numeric ID column, and skips broken records.
    """

    print("\nUsing manual CSV repair reader...")
    print("This will skip corrupted records and recover valid rows.")

    rows = []
    bad_rows = 0

    record_start_pattern = re.compile(r"^\d+,")

    def parse_record(record_text):
        try:
            parsed = next(
                csv.reader(
                    [record_text],
                    quotechar='"',
                    doublequote=True,
                    escapechar="\\"
                )
            )

            # Expected columns:
            # 0 = Unnamed index
            # 1 = title
            # 2 = text
            # 3 = label
            if len(parsed) >= 4:
                title = parsed[1]
                text = parsed[2]
                label = parsed[3]

                return {
                    "title": title,
                    "text": text,
                    "label": label
                }

        except Exception:
            return None

        return None

    current_record = ""

    max_record_chars = 5_000_000

    with open(DATA_PATH, "r", encoding="utf-8", errors="replace", newline="") as file:
        header = file.readline()

        for line_number, line in enumerate(file, start=2):
            if record_start_pattern.match(line) and current_record:
                parsed_record = parse_record(current_record)

                if parsed_record is not None:
                    rows.append(parsed_record)
                else:
                    bad_rows += 1

                current_record = line

            else:
                current_record += line

                if len(current_record) > max_record_chars:
                    bad_rows += 1
                    current_record = ""

        if current_record:
            parsed_record = parse_record(current_record)

            if parsed_record is not None:
                rows.append(parsed_record)
            else:
                bad_rows += 1

    print(f"Manual repair completed.")
    print(f"Recovered good rows: {len(rows)}")
    print(f"Skipped corrupted rows: {bad_rows}")

    if len(rows) == 0:
        raise ValueError("Manual CSV repair failed. No valid rows were recovered.")

    return pd.DataFrame(rows)


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put your CSV file here: data/fake_news_merged.csv"
        )

    print("Loading dataset...")

    try:
        df = try_pandas_read_csv()

    except Exception as e:
        print("\nNormal CSV reading failed.")
        print("Reason:", e)
        df = manual_csv_repair_reader()

    df.columns = [clean_column_name(c) for c in df.columns]

    print("Columns found:", df.columns.tolist())

    required_cols = ["title", "text", "label"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found.\n"
                f"Available columns: {df.columns.tolist()}"
            )

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["combined_text"] = (
        df["title"].astype(str)
        + " "
        + df["text"].astype(str)
    )

    df["label"] = df["label"].apply(encode_label)

    before_label_cleaning = len(df)
    df = df.dropna(subset=["label"])
    after_label_cleaning = len(df)

    print("Rows removed because of invalid labels:", before_label_cleaning - after_label_cleaning)

    df["label"] = df["label"].astype(int)

    before_duplicate_cleaning = len(df)
    df = df.drop_duplicates(subset=["combined_text", "label"])
    after_duplicate_cleaning = len(df)

    print("Duplicate rows removed:", before_duplicate_cleaning - after_duplicate_cleaning)

    df = df[df["combined_text"].str.len() > 20]

    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)

    print("\nFinal dataset shape:", df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    if df["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both Fake and Real labels after cleaning."
        )

    return df


# ============================================================
# Dataset Class
# ============================================================

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# Training and Evaluation Functions
# ============================================================

def train_one_epoch(model, data_loader, optimizer, scheduler):
    model.train()

    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(data_loader)


def evaluate_model(model, data_loader):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    results = {
        "loss": total_loss / len(data_loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return results, all_labels, all_predictions


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    set_seed(RANDOM_STATE)

    print("=" * 70)
    print("LLM/Transformer-Based Fake News Detection")
    print("=" * 70)
    print("Device:", DEVICE)

    df = load_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["combined_text"],
        df["label"],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print("\nData Split:")
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = FakeNewsDataset(
        texts=X_train,
        labels=y_train,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    val_dataset = FakeNewsDataset(
        texts=X_val,
        labels=y_val,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    test_dataset = FakeNewsDataset(
        texts=X_test,
        labels=y_test,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nLoading DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_training_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps
    )

    best_val_f1 = -1

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler
        )

        val_results, _, _ = evaluate_model(
            model=model,
            data_loader=val_loader
        )

        print("Training Loss:", round(train_loss, 4))
        print("Validation Results:", val_results)

        if val_results["f1_score"] > best_val_f1:
            best_val_f1 = val_results["f1_score"]

            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

            print("Best model saved.")

    print("\nLoading best model for final test evaluation...")

    best_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    best_model = best_model.to(DEVICE)

    test_results, test_labels, test_predictions = evaluate_model(
        model=best_model,
        data_loader=test_loader
    )

    print("\nFinal Test Results:")
    print(test_results)

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))

    label_mapping = {
        "0": "Fake",
        "1": "Real"
    }

    with open(SAVE_DIR / "label_mapping.json", "w") as file:
        json.dump(label_mapping, file, indent=4)

    joblib.dump(
        {
            "max_length": MAX_LENGTH,
            "model_name": MODEL_NAME,
            "test_results": test_results,
            "max_rows": MAX_ROWS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        },
        SAVE_DIR / "training_info.pkl"
    )

    print("\nTraining completed successfully.")
    print("Saved model folder:", SAVE_DIR)


if __name__ == "__main__":
    main()