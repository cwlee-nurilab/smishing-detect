from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.inference.inference import Inference
from src.utils.load_config import ConfigLoader
from src.utils.io import read_jsonl, write_jsonl


if __name__ == "__main__":
    model = Inference("beomi/KcELECTRA-base")
    path_loader = ConfigLoader("./src/config/path.yaml")

    testsets = read_jsonl(path_loader.get("test.test_jsonl"))
    output_dir = path_loader.get("test.output_dir")

    labels = []
    preds = []
    results =[]
    errors = []

    for item in tqdm(testsets):
        pred_item = model.predict(item["full_text"])

        labels.append(int(item["label"]))
        preds.append(int(pred_item["label"]))

        result = {
            "pred_label":  pred_item["label"],
            **item,
            "probabilities": pred_item["probability"],
        }

        if int(item["label"]) != int(pred_item["label"]):
            errors.append(result)

        results.append(result)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    
    print('accuracy:', acc)
    print('f1:', f1)
    print('precision:', precision)
    print('recall:', recall)
    print('auroc:', auc)

    write_jsonl(f"{output_dir}/KcELECTRA-base.jsonl", data=results)
    write_jsonl(f"{output_dir}/KcELECTRA-base_error.jsonl", data=errors)

    