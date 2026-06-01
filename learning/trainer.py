"""ModelTrainer — trains models from datasets built by DatasetBuilder.

Supports two training modes:

1. **classifier**  — fine-tune a DistilBERT classifier (same pattern as
   the existing train_intent.py but reusable).  Good for action
   prediction: given (context, goal) → predict best action type.

2. **reasoning** (stub) — future slot for fine-tuning a causal LM on
   full (goal → action chain) episodes so the reasoning model improves
   from experience.

All trained artifacts land in memory_store/models/<run_name>/.
"""

import json
import os
import time
from pathlib import Path


class ModelTrainer:
    """Trains models from datasets produced by DatasetBuilder."""

    def __init__(self, store_dir: str = "memory_store"):
        self._models_dir = Path(store_dir) / "models"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._last_run: dict | None = None

    # ── Public API ───────────────────────────────────────────────

    def train_action_classifier(
        self,
        dataset_path: str,
        run_name: str = "",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        text_col: str = "context",
        label_col: str = "action",
    ) -> dict:
        """Fine-tune a DistilBERT classifier to predict actions from context.

        Args:
            dataset_path: Path to CSV with *text_col* and *label_col* columns.
            run_name:     Name for this training run (auto-generated if empty).
            epochs:       Number of training epochs.
            batch_size:   Per-device batch size.
            learning_rate: Learning rate.
            text_col:     CSV column used as input text.
            label_col:    CSV column used as the label.

        Returns:
            dict with keys: model_dir, eval_accuracy, num_labels, rows
        """
        # Lazy import so the module loads fast when training isn't needed
        import numpy as np
        from datasets import load_dataset
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )

        if not run_name:
            run_name = f"action_clf_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir = str(self._models_dir / run_name)

        # Load CSV
        ds = load_dataset("csv", data_files=dataset_path)["train"]
        labels = sorted(set(ds[label_col]))
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}
        print(f"[ModelTrainer] Labels ({len(labels)}): {labels}")

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

        def preprocess(example):
            tok = tokenizer(
                example[text_col],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            tok["label"] = label2id[example[label_col]]
            return tok

        ds = ds.map(preprocess, remove_columns=ds.column_names)
        split = ds.train_test_split(test_size=0.15, seed=42, shuffle=True)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        def compute_metrics(eval_pred):
            logits, labs = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": float((preds == labs).mean())}

        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=2,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            compute_metrics=compute_metrics,
        )

        print(f"[ModelTrainer] Training '{run_name}' — {epochs} epochs …")
        trainer.train()

        eval_results = trainer.evaluate()
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        # Save run metadata
        meta = {
            "run_name": run_name,
            "dataset": dataset_path,
            "labels": labels,
            "num_labels": len(labels),
            "epochs": epochs,
            "eval_accuracy": eval_results.get("eval_accuracy", 0.0),
            "eval_loss": eval_results.get("eval_loss", 0.0),
            "rows": len(ds),
            "model_dir": out_dir,
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._last_run = meta
        print(f"[ModelTrainer] Done — accuracy {meta['eval_accuracy']:.3f} → {out_dir}")
        return meta

    def train_reasoning_model(
        self,
        dataset_path: str,
        run_name: str = "",
        **kwargs,
    ) -> dict:
        """Fine-tune a reasoning model on episode chains (future).

        This will accept JSONL datasets produced by
        DatasetBuilder.build_reasoning_dataset() and fine-tune a causal
        language model to predict action sequences given a goal + context.

        Currently returns a placeholder — the training loop will be
        implemented when a suitable base model is chosen.
        """
        if not run_name:
            run_name = f"reasoning_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir = str(self._models_dir / run_name)
        os.makedirs(out_dir, exist_ok=True)

        # Count episodes in dataset
        episode_count = 0
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episode_count += 1

        meta = {
            "run_name": run_name,
            "dataset": dataset_path,
            "episodes": episode_count,
            "status": "pending",
            "model_dir": out_dir,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Reasoning model training not yet implemented. "
                    "Dataset prepared and validated.",
        }
        with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._last_run = meta
        print(f"[ModelTrainer] Reasoning stub — {episode_count} episodes logged → {out_dir}")
        return meta

    def train_correction_classifier(
        self,
        dataset_path: str,
        run_name: str = "",
        epochs: int = 6,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
    ) -> dict:
        """Train a classifier to map user correction text to corrected intent label."""
        if not run_name:
            run_name = f"correction_clf_{time.strftime('%Y%m%d_%H%M%S')}"
        return self.train_action_classifier(
            dataset_path=dataset_path,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            text_col="input_text",
            label_col="corrected_intent",
        )

    def list_runs(self) -> list[dict]:
        """Return metadata for all training runs."""
        runs = []
        for d in sorted(self._models_dir.iterdir()):
            meta_file = d / "run_meta.json"
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    runs.append(json.load(f))
        return runs

    @property
    def last_run(self) -> dict | None:
        return self._last_run
