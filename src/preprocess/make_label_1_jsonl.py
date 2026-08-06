import os
import re
from tqdm import tqdm
from pathlib import Path

from src.utils.load_config import ConfigLoader
from src.utils.io import read_jsonl, write_jsonl


def make_label_1_jsonl():
    path_loader = ConfigLoader("./src/config/path.yaml")
    raw_base_dir = path_loader.get("raw.base_dir")
    raw_label_1_jsonl_name = path_loader.get("raw.label_1_jsonl")

    raw_label_1_jsonl_path = os.path.join(raw_base_dir, raw_label_1_jsonl_name)

    # 기존 파일 불러오기
    raw_data = read_jsonl(str(raw_label_1_jsonl_path))
    final_data = []

    for item in tqdm(raw_data):
        # 정규식으로 url 추출
        urls = re.findall(r"https?://[^\s]+", item["text"])
        if not urls:
            urls = re.findall(
                r"\b(?:https?://)?(?:[A-Za-z0-9가-힣-\?]+\.)+[A-Za-z가-힣-\?]{2,}(?:/[^\s]*)?",
                item["text"],
                )

        # jsonl 형태로 맞춤, label 1인경우 detail 키 추가
        final_data.append({
            "label": "1",
            "full_text": item["text"],
            "urls": urls,
            "detail": item["label"],
        })

    processed_base_dir = path_loader.get("processed.base_dir")
    processed_label_1_jsonl_name = path_loader.get("processed.label_1_jsonl")

    processed_label_1_jsonl_path = os.path.join(processed_base_dir, processed_label_1_jsonl_name)
    write_jsonl(str(processed_label_1_jsonl_path), final_data)