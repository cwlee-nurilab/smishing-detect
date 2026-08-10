import abc

from src.utils.load_config import ConfigLoader


class TrainerBase(abc.ABC):
    def __init__(self, mode: str):
        self.mode = mode

        self.path_loader = ConfigLoader("./src/config/path.yaml")
        self.config_loader = ConfigLoader("./src/config/model.yaml")

        self.model_name = self.config_loader.get(f"{self.mode}.model_name")
        self.device = self.config_loader.get(f"{self.mode}.device")

        self.train_jsonl = self.path_loader.get(f"train.train_jsonl")
        self.val_jsonl = self.path_loader.get(f"val.val_jsonl")
        
        self.checkpoint_dir = self.config_loader.get(f"{self.mode}.train.output_dir")


    # helpers
    def _get_train_config(self, key: str):
        return self.config_loader.get(f"{self.mode}.train.{key}")


    @abc.abstractmethod
    def _get_model_and_tokenizer(self):
        pass


    @abc.abstractmethod
    def _data_loader(self):
        pass


    # public method
    @abc.abstractmethod
    def train(self):
        pass


