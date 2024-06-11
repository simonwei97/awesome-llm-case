import argparse
import datetime
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, random_split
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")

    logger.debug("there are %d GPU(s) available." % torch.cuda.device_count())

    logger.debug("use GPU info: {}", torch.cuda.get_device_name(0))
else:
    logger.debug("No GPU available, using the CPU instead.")

    device = torch.device("cpu")

MAX_LEN = 512  # 根据训练语料的最大长度决定
BERT_MODEL_NAME = "/home/weixiaopeng1/bert_train/models/bert-base-chinese"
TRAIN_EPOCHS = int(args.epoch) if args.epoch is not None else 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
ADAM_EPSILON = 1e-8
SEED_VAL = 42

output_dir = "./pytorch_bert_model/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger.debug("start train model: {}", datetime.datetime.now())
start_time = time.time()

# 1. 加载和预处理数据集
data = pd.read_csv("waimai_10k.csv")

# 2. 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

reviews = data.review.values
labels = data.label.values
class_names = data.label.unique()

# tokenize 所有的文本
input_ids = []
attention_masks = []

for review in reviews:
    encoded_dict = tokenizer.encode_plus(
        review,
        add_special_tokens=True,  # 添加特殊标记, [CLS] 和 [SEP]
        max_length=MAX_LEN,  # 最大长度
        return_token_type_ids=True,  # 分句ids, 返回 token_type_ids
        pad_to_max_length=True,  # 填充到最大长度
        return_attention_mask=True,  # 返回 attention_mask
        return_tensors="pt",  # 返回 PyTorch 张量
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict["input_ids"])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict["attention_mask"])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)


# **************  DataLoader **************

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
logger.debug("{:>5,} training samples".format(train_size))
logger.debug("{:>5,} eval samples".format(eval_size))

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE,
)

eval_dataloader = DataLoader(
    eval_dataset,
    sampler=RandomSampler(eval_dataset),
    batch_size=BATCH_SIZE,
)

# **************** Train ****************

bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME,
    num_labels=len(class_names),
    output_attentions=False,
    output_hidden_states=False,
)

bert_model.to(device)


# *************** Optimizer ************

optimizer = AdamW(
    bert_model.parameters(),
    lr=LEARNING_RATE,
    eps=ADAM_EPSILON,
)


total_steps = len(train_dataloader) * TRAIN_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # Default value in run_glue.py
    num_training_steps=total_steps,
)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, TRAIN_EPOCHS):

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    logger.debug("======== Epoch {:} / {:} ========".format(epoch_i + 1, TRAIN_EPOCHS))
    logger.debug("Training...")
    t0 = time.time()
    total_train_loss = 0

    bert_model.train()

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        output = bert_model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss = output.loss
        total_train_loss += loss.item()

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    logger.debug("  Average training loss: {0:.2f}".format(avg_train_loss))
    logger.debug("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    logger.debug("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    bert_model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    best_eval_accuracy = 0

    # Evaluate data for one epoch
    for batch in eval_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            output = bert_model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

        loss = output.loss
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = output.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(eval_dataloader)
    logger.debug("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(eval_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    if avg_val_accuracy > best_eval_accuracy:

        # torch.save(bert_model, "bert_best_model")
        best_eval_accuracy = avg_val_accuracy
        logger.debug("saving model to %s" % output_dir)
        model_to_save = (
            bert_model.module if hasattr(bert_model, "module") else bert_model
        )
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    logger.debug("  Validation Loss: {0:.2f}".format(avg_val_loss))
    logger.debug("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

logger.debug("")
logger.debug("Training complete!")

logger.debug(
    "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
)
