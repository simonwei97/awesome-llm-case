import argparse
import warnings

from loguru import logger
from transformers import BertForSequenceClassification, BertTokenizer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str)
args = parser.parse_args()


# load model and tokenizer
# !!! please modify this path
model_name = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

text = args.text
logger.debug(f"predict text: {text}")

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

# get predict output
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

predicted_class_id = outputs.logits.argmax().item()
predicted_label = id2label[predicted_class_id]

logger.debug(f"predict output：{predicted_class_id} -> {predicted_label}")
