import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config,  AutoTokenizer, AutoModelWithLMHead

logging.basicConfig(filename='qg_incremental.log', filemode='w', level=logging.DEBUG)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.debug('Using device:' + str(device))

train = pd.read_csv('~/qg_dataset/hotpot_train_fullcontext_v1.1.csv')
# train = train[40001:]
valid = pd.read_csv('~/qg_dataset/hotpot_dev_distractor_fullcontext_v1.csv')
# valid = valid[:5000]


# Calculating token length



# import matplotlib.pyplot as plt
# import seaborn as sns

# token_lens = []
# for txt in train.text:
#   tokens = tokenizer.encode(txt, max_length=512)
#   token_lens.append(len(tokens))

# sns.distplot(token_lens)
# plt.xlim([0, 512]);
# plt.xlabel('Token lengths');



PRETRAINED_MODEL = 't5-base'
DIR = "question_generator/"
BATCH_SIZE = 1
SEQ_LENGTH = 600

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<answer>', '<context>']}
)
class QGDataset(Dataset):
    def __init__(self, csv):
        self.df = csv

    def __len__(self):
         return len(self.df)

    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx, 1:]       

        encoded_text = tokenizer(
            row['text'], 
            padding=True, 
            max_length=SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        encoded_text['input_ids'] = torch.squeeze(encoded_text['input_ids'])
        encoded_text['attention_mask'] = torch.squeeze(encoded_text['attention_mask'])

        encoded_question = tokenizer(
            row['question'],
            padding=True,
            max_length=SEQ_LENGTH,
            truncation=True,
            return_tensors='pt'
        )
        encoded_question['input_ids'] = torch.squeeze(encoded_question['input_ids'])

        return (encoded_text.to(device), encoded_question.to(device))

train_set = QGDataset(train)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_set = QGDataset(valid)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)


LR = 0.001
EPOCHS = 20
LOG_INTERVAL = 5000

#FIRST TRAINING
config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
model = T5ForConditionalGeneration(config).from_pretrained(PRETRAINED_MODEL)
model.resize_token_embeddings(len(tokenizer)) # to account for new special tokens
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


# Both last model and model have been saved -> then the next set of lines

# model = torch.load("/home2/samyak.ja/qg_dataset/model_hotpot_last.pth")
# m1 = torch.load("/home2/samyak.ja/qg_dataset/model_hotpot.pth")
# model.load_state_dict(m1['model_state_dict'])
# model = model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# optimizer.load_state_dict(m1['optimizer_state_dict'])


SAVED_MODEL_PATH = "/home2/samyak.ja/qg_dataset/model_hotpot.pth"
TEMP_SAVE_PATH = "/home2/samyak.ja/qg_dataset/model_hotpot.pth"

def train(epoch, best_val_loss):
    model.train()
    total_loss = 0.
    for batch_index, batch in enumerate(train_loader):
        data, target = batch
        optimizer.zero_grad()
        masked_labels = mask_label_padding(target['input_ids'])
        output = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            labels=masked_labels
        )
        loss = output[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch_index % LOG_INTERVAL == 0 and batch_index > 0:
            cur_loss = total_loss / LOG_INTERVAL
            logging.debug(str(epoch) + " " +str(batch_index)+ " " + str(cur_loss))
            save(
                TEMP_SAVE_PATH,
                epoch, 
                model.state_dict(), 
                optimizer.state_dict(), 
                best_val_loss
            )
            total_loss = 0

def evaluate(eval_model, data_loader):
    eval_model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            data, target = batch
            masked_labels = mask_label_padding(target['input_ids'])
            output = eval_model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                labels=masked_labels
            )
            total_loss += output[0].item()
    return total_loss / len(data_loader)

def mask_label_padding(labels):
    MASK_ID = -100
    labels[labels==tokenizer.pad_token_id] = MASK_ID
    return labels

def save(path, epoch, model_state_dict, optimizer_state_dict, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'best_loss': loss,
            }, path)

def load(path):
    return torch.load(path)


best_val_loss = float("inf")
best_model = None

val_loss = evaluate(model, valid_loader)
logging.debug('Before training'+ str(val_loss))

for epoch in range(1, EPOCHS + 1):

    train(epoch, best_val_loss)
    val_loss = evaluate(model, valid_loader)
    logging.debug(str(epoch) + " " + str(val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        save(SAVED_MODEL_PATH, epoch, model.state_dict(), optimizer.state_dict(), best_val_loss)
        logging.debug("| Model saved.")


torch.save(model, "/home2/samyak.ja/qg_dataset/model_hotpot_last.pth")
md2 = torch.load("/home2/samyak.ja/qg_dataset/model_hotpot_last.pth")

val_loss = evaluate(md2, valid_loader)
logging.debug('Post training - final model' + str(val_loss))


def inference(review_text, model, device):
    encoded_text = tokenizer(
            review_text, 
            padding=True, 
            max_length=SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(device)

    input_ids = encoded_text['input_ids']
    with torch.no_grad():
        output = model.generate(input_ids)
    decoded_string = tokenizer.decode(output[0], skip_special_tokens=True)
    logging.debug("Decoded string" + decoded_string)



review_text = "<answer> a fusional language <context> Typologically, Estonian represents a transitional form from an agglutinating language to a fusional language. The canonical word order is SVO (subject–verb–object)."

inference(review_text, md2, device)
