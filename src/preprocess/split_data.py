import os
import random

from src.utils.io import read_jsonl, write_jsonl
from src.utils.load_config import ConfigLoader


def _shuffle_and_split(data, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.Random(seed).shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    train = data[:train_end]
    valid = data[train_end:valid_end]
    test = data[valid_end:]

    return train, valid, test


def split_data():
    print("[START] split_data")
    path_loader = ConfigLoader("./src/config/path.yaml")
    processed_base_dir = path_loader.get("processed.base_dir")
    label_0_jsonl_name = path_loader.get("processed.label_0_jsonl")
    label_1_jsonl_name = path_loader.get("processed.label_1_jsonl")

    label_0_path = os.path.join(processed_base_dir, label_0_jsonl_name)
    label_1_path = os.path.join(processed_base_dir, label_1_jsonl_name)

    li_label_0 = read_jsonl(label_0_path)
    li_label_1 = read_jsonl(label_1_path)

    train_0, valid_0, test_0 = _shuffle_and_split(li_label_0)
    train_1, valid_1, test_1 = _shuffle_and_split(li_label_1)

    train = train_0 + train_1
    valid = valid_0 + valid_1
    test = test_0 + test_1

    processed_base_dir = path_loader.get("processed.base_dir")
    label_0_jsonl_name = path_loader.get("processed.label_0_jsonl")
    label_1_jsonl_name = path_loader.get("processed.label_1_jsonl")

    train_jsonl = path_loader.get("train.train_jsonl")
    val_jsonl = path_loader.get("val.val_jsonl")
    test_jsonl = path_loader.get("test.test_jsonl")

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    write_jsonl(train_jsonl, train)
    write_jsonl(val_jsonl, valid)
    write_jsonl(test_jsonl, test)
    
    print("[DONE] split_data")


    return train, valid, test

