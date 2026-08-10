from src.train.bert_trainer import BERTTrainer


if __name__ == "__main__":
    bert_trainer = BERTTrainer(mode="beomi/KcELECTRA-base")
    bert_trainer.train()