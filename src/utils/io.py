import msoffcrypto
import io
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm



def read_xlsx(path: str, password: str = None) -> pd.DataFrame:
    if password:
        decrypted = io.BytesIO()

        with open(path, "rb") as f:
            file = msoffcrypto.OfficeFile(f)
            file.load_key(password=password)
            file.decrypt(decrypted)

        df = pd.read_excel(decrypted, engine="openpyxl")

    else:
        df = pd.read_excel(path, engine="openpyxl")


    return df


def read_csv(file_path):
    return pd.read_csv(file_path)


def read_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_md(file_name:str, **kwargs):
    template = Path(file_name).read_text(encoding="utf-8")
    return template.format(**kwargs)


def write_jsonl(path: str, data: list):
    with open(path, "w", encoding="utf-8") as f:
        for item in tqdm(data):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"저장 완료: {path}")