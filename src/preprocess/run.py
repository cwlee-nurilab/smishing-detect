from src.preprocess.make_label_1_jsonl import make_label_1_jsonl
from src.preprocess.make_label_0_jsonl import make_label_0_jsonl
from src.preprocess.split_data import split_data

if __name__ == "__main__":
    make_label_0_jsonl()
    make_label_1_jsonl()
    split_data()