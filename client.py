import argparse
import onnxruntime as ort
import json
import numpy as np
from config import *
from transformers import BertTokenizer

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, required=True)
args = parser.parse_args()
text = args.input_text

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model_path = "review_classifier_model.onnx"
session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"],
)
encoding = tokenizer(
    text,
    return_tensors="np",
    max_length=max_length,
    padding="max_length",
    truncation=True,
)
input_ids = np.array(encoding["input_ids"], dtype=np.int64)
attention_mask = np.array(encoding["attention_mask"], dtype=np.int64)
outputs = session.run(None, {"input": input_ids, "attention_mask": attention_mask})
predicted_class = np.argmax(outputs[0][0])
out = "safe" if predicted_class == 1 else "not-safe"

out = {"text": text, "prediction": out}

with open("output.json", "w") as f:
    json.dump(out, f)
