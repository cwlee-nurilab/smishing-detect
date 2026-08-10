import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.load_config import ConfigLoader


class Inference:
    def __init__(self, mode):
        self.mode = mode

        self.path_loader = ConfigLoader("./src/config/path.yaml")
        self.config_loader = ConfigLoader("./src/config/model.yaml")

        self.checkpoint_dir = self.config_loader.get(f"{self.mode}.train.output_dir")
        self.model_dir = f"{self.checkpoint_dir}/best_model"
        self.device = 0

        self._get_model_and_tokenizer()


    def _get_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            verbose=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )
        self.model.to(self.device)


    @torch.no_grad()
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

        return {
            "label": pred,
            "probability": probs[0, pred].item(),
            "probabilities": probs[0].cpu().tolist(),
        }