import abc
from tempfile import template

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from src.utils.io import read_jsonl
from src.utils.load_config import ConfigLoader


class SFTTrainerBase(abc.ABC):
    def __init__(self, mode: str):
        self.mode = mode

        self.path_loader = ConfigLoader("./src/config/path.yaml")
        self.config_loader = ConfigLoader("./src/config/model.yaml")

        self.model_name = self.config_loader.get(f"{self.mode}.model_name")
        self.device = self.config_loader.get(f"{self.mode}.device")

        self.input_jsonl = self.path_loader.get(f"{self.mode}.train.train_jsonl")
        self.checkpoint_dir = self.checkpoint_dir.get(f"{self.mode}.train.checkpoint_dir")


    # helpers
    def _get_train_config(self, key: str):
        return self.config_loader.get(f"{self.mode}.train.train_jsonl")


    def _get_model_and_tokenizer(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer


    @abc.abstractmethods
    def _get_lora_cfg(self) -> LoraConfig:
        pass


    @abc.abstractmethods
    def _get_training_args(self) -> SFTConfig:
        pass


    # data_loader
    @abc.abstractmethods
    def _data_loader(self) -> Dataset:
        pass


    # public method
    def train(self):
        dataset = self._data_loader()
        model, tokenizer = self._get_model_and_tokenizer()

        lora_cfg = self._get_lora_cfg()
        training_args = self._get_training_args()

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            peft_config=lora_cfg,
            processing_class=tokenizer
        )

        trainer.train()


