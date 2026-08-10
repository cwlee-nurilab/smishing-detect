import abc
from tempfile import template

import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    Trainer,
    logging
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.train.base import TrainerBase
from src.train.loss_logger import LossLoggerCallback


class BERTTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.label2id = {
            "normal": 0,
            "smishing": 1,
        }
        self.id2label = {
            0: "normal",
            1: "smishing",
        }

    # helpers
    def _get_model_and_tokenizer(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self._get_train_config("num_labels"),
            id2label=self.id2label,
            label2id=self.label2id
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        return model, tokenizer



    # data_loader
    def _data_loader(self):
        dataset = load_dataset(
            "json",
            data_files={
                "train": self.train_jsonl,
                "val": self.val_jsonl,
            },
        )

        return dataset


    # preprocessor
    def _preprocess_data(self, examples, tokenizer):
        tokenized = tokenizer(
            examples["full_text"],
            truncation=True,
            padding="max_length",
            max_length=512, # 99.95% 커버 가능
        )

        tokenized["labels"] = [int(x) for x in examples["label"]]

        return tokenized


    # Trainer compute_metrics
    def _compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auroc': auc
        }


    # optuna
    def _hp_space(self, trial):
        return {
        "num_train_epochs": trial.suggest_int(
            name="num_train_epochs",
            low=int(self._get_train_config("num_train_epochs.low")),
            high=int(self._get_train_config("num_train_epochs.high")),
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(self._get_train_config("learning_rate.low")),
            float(self._get_train_config("learning_rate.high")),
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            float(self._get_train_config("weight_decay.low")),
            float(self._get_train_config("weight_decay.high"))
        ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio",
            float(self._get_train_config("warmup_ratio.low")),
            float(self._get_train_config("warmup_ratio.high"))
        ),
    }


    # public method
    def train(self):
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self._get_train_config("num_labels"),
                id2label=self.id2label,
                label2id=self.label2id
            )
        _, tokenizer = self._get_model_and_tokenizer()

        dataset = self._data_loader()
        dataset = dataset.map(
            self._preprocess_data,
            batched=True,
            remove_columns=dataset["train"].column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
            },
        )
        dataset.set_format('torch')

        training_args = TrainingArguments(
            # 고정
            logging_dir='./logs',
            logging_steps=1000,
            do_train=True,
            do_eval=True,
            bf16=True,
            fp16=False,   
            metric_for_best_model="f1",
            load_best_model_at_end=True,
            greater_is_better=True,
            save_total_limit=2,
            # yaml
            output_dir=self._get_train_config("output_dir"),
            eval_strategy=self._get_train_config("eval_strategy"),
            save_strategy=self._get_train_config("save_strategy"),
            per_device_train_batch_size=self._get_train_config("per_device_train_batch_size"),
            per_device_eval_batch_size=self._get_train_config("per_device_eval_batch_size"),
        )

        trainer = Trainer(
            model_init=model_init,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            args=training_args,
            compute_metrics=self._compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[
                LossLoggerCallback("./logs")
            ]
        )

        best_run = trainer.hyperparameter_search(
            backend="optuna",
            hp_space=self._hp_space,
            direction="maximize",
            compute_objective=lambda metrics: metrics["eval_f1"],
            n_trials=30,
        )


        # 베스트모델 재학습
        for k, v in best_run.hyperparameters.items():
            setattr(trainer.args, k, v)

        trainer.train()


        # 최종 베스트 모델 및 토크나이저 저장
        best_model_path = f"{self.checkpoint_dir}/best_model"
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)

        print(f"Best model saved to {best_model_path}")


