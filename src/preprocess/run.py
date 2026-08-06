from src.preprocess.make_label_1_jsonl import make_label_1_jsonl
from src.preprocess.make_label_0_jsonl import make_label_0_jsonl

if __name__ == "__main__":
    make_label_0_jsonl()
    make_label_1_jsonl()