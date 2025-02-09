from typing import TypedDict
import torch
from config import *
from transformers import BertTokenizer
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    BatchTextInput, 
    BatchTextResponse, 
    EnumParameterDescriptor, 
    EnumVal, 
    InputSchema, 
    InputType, 
    ParameterSchema, 
    ResponseBody, 
    TaskSchema, 
    TextResponse
)

## ORIGINAL
# from App_dangerrousness import BERTClassifiers

## ONNX
import onnxruntime as ort
import numpy as np

# configure UI elements
def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Input Text",
        input_type=InputType.BATCHTEXT,
    )
    return TaskSchema(inputs=[input_schema], parameters=[])

class Inputs(TypedDict):
    input_dataset: BatchTextInput

class Parameters(TypedDict):
    pass

# Initialize Flask app
server = MLServer(__name__)
server.add_app_metadata(
    name="App Dangerousness Classifier",
    author="UMass Rescue",
    version="0.1.0",
    info=load_file_as_string("README.md"),
)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

## ORIGINAL
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_model(bert_model_name, num_classes, batch_size, learning_rate, model_path, device):
#     model = BERTClassifier(bert_model_name, num_classes, batch_size, learning_rate).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()  # Set the model to evaluation mode
#     return model

# model = load_model(
#     bert_model_name, num_classes, batch_size, learning_rate, model_path, device
# )

## ONNX
model_path = "review_classifier_model.onnx"
session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

def predict_text_class(text):
    encoding = tokenizer(
        text,
        return_tensors="np",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    ## ORIGINAL CODE
    # input_ids = torch.tensor(encoding["input_ids"]).to(device)
    # attention_mask = torch.tensor(encoding["attention_mask"]).to(device)

    # with torch.no_grad():
    #     import pdb; pdb.set_trace()
    #     # to get the onnx model
    #     # torch.onnx.export(model, text, "model.onnx", export_params = True, opset_version = 16, do_constant_folding = True, input_names = ["input"], output_names = ["output"], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
    #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # _, predicted_class = torch.max(outputs, dim=1)

    ## ONNX
    input_ids = np.array(encoding["input_ids"], dtype = np.int64)
    attention_mask = np.array(encoding["attention_mask"], dtype = np.int64)
    outputs = session.run(None, {"input": input_ids, "attention_mask": attention_mask})
    predicted_class = np.argmax(outputs[0][0])
    return predicted_class

@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def predict(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    outputs = []
    for text_input in inputs['input_dataset'].texts:
        raw_text = text_input.text
        pred_class = predict_text_class(raw_text)
        predicted_class = "Safe" if pred_class == 1 else "not-safe"
        outputs.append(TextResponse(value=predicted_class, title=raw_text))
    return ResponseBody(root=BatchTextResponse(texts=outputs))

# Start the Flask app
if __name__ == "__main__":
    server.run()