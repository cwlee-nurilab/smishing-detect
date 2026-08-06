import re
import os
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

from huggingface_hub import login
from datasets import load_dataset

import pandas as pd
from src.utils.load_config import ConfigLoader
from src.utils.io import read_xlsx, write_jsonl, read_csv
from src.preprocess.build_dummy_data import build_dummy_data

def _get_urls(full_text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s]+", full_text)
    if not urls:
        urls = re.findall(
            r"\b(?:https?://)?(?:[A-Za-z0-9가-힣-\?]+\.)+[A-Za-z가-힣-\?]{2,}(?:/[^\s]*)?",
            full_text,
            )

    return urls


def _xlsxs_to_dict_list(label_0_dir_path: str, password: str = None) -> list[dict]:
    dfs = []
    xlsxs = list(Path(label_0_dir_path).glob("*.xlsx"))

    for xlsx in xlsxs:
        df = read_xlsx(str(xlsx), password)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    li_dict = []
    for _, row in tqdm(merged_df.iterrows(), 
                        total=len(merged_df),
                        desc="Processing xlsxs"):
        urls = _get_urls(row["SMS 내용"])
        urls.append(str(row["검색URL"]))
        urls = list(set(urls))

        li_dict.append({
            "label": "0",
            "full_text": str(row["SMS 내용"]),
            "urls": urls,
            "detail": "kisa"
        })

    return li_dict


def _load_hf_datasets(token: str) -> list[dict]:
    login(token=token)
    ds = load_dataset("meal-bbang/Korean_message")

    li_dict = []
    for item in ds["train"]:
        if int(item["class"])==1:
            urls = _get_urls(item["content"])
            li_dict.append({
                            "label": "0",
                            "full_text": item["content"],
                            "urls": urls,
                            "detail": "meal-bbang/Korean_message"
                        })

    return li_dict


def _load_hf_csv(path:str):
    '''
    출처: https://huggingface.co/datasets/jmjmjm3/kor-smishing-message/tree/main
    '''
    df = read_csv(path)

    li_dict = []
    for _, item in tqdm(df.iterrows(), 
                        total=len(df),
                        desc="Processing csv"):
        if item["label"]=="정상":
            urls = _get_urls(item["content"])
            li_dict.append({
                "label": "0",
                "full_text": item["content"],
                "urls": urls,
                "detail": "jmjmjm3/kor-smishing-message"
            })

    return li_dict


def make_label_0_jsonl():
    load_dotenv()
    PASSWORD = os.getenv("PASSWORD", None)
    HF_TOKEN = os.getenv("HF_TOKEN", None)

    path_loader = ConfigLoader("./src/config/path.yaml")
    raw_base_dir = path_loader.get("raw.base_dir")
    label_0_dir_name = path_loader.get("raw.label_0_dir")
    label_0_csv_name = path_loader.get("raw.label_0_csv")

    label_0_dir_path = os.path.join(raw_base_dir, label_0_dir_name)
    label_0_csv_path = os.path.join(raw_base_dir, label_0_csv_name)

    # xlsx 읽어온 뒤 concat
    # dict list 형태로 변경
    '''
    내부 dict 형태
    {
        "label": "0",
        "full_text": str,
        "urls": list[str],
        "detail: str # 출처
    }
    '''
    kisa_data = _xlsxs_to_dict_list(label_0_dir_path, PASSWORD)
    hf_datasets_data = _load_hf_datasets(HF_TOKEN)
    hf_csv_data = _load_hf_csv(label_0_csv_path)

    final_results = kisa_data + hf_datasets_data + hf_csv_data

    # 더미 생성
    dummy_data = build_dummy_data(label_0_dir_path)
    for full_text in dummy_data:
        final_results.append({
            "label": "0",
            "full_text": full_text,
            "urls": _get_urls(full_text),
            "detail": "dummy"
        })

    # jsonl로 저장
    processed_base_dir = path_loader.get("processed.base_dir")
    label_0_jsonl_name = path_loader.get("processed.label_0_jsonl")

    label_0_jsonl_path = os.path.join(processed_base_dir, label_0_jsonl_name)
    write_jsonl(str(label_0_jsonl_path), final_results)
