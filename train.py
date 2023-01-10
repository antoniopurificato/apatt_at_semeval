#Import
import pandas as pd
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import pickle
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import torch
import numpy as np

import pytorch_lightning as pl
from transformers import get_constant_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

#Parser
parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, choices = ["Bert", "RoBERTa", "XLNet","DeBERTa","alBERT"], help='Choice of the model')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--language', type=str, help='Language')
parser.add_argument('--threshold', type=float, help='Value of the threshold')
parser.add_argument('--mode', type=str, help='online or offline')
args = parser.parse_args()
model_name = args.models
EPOCHS = args.epochs
LANGUAGE = args.language
THRESHOLD = args.threshold
os.environ['WANDB_MODE'] = args.mode


#Useful settings
PATH = '/home/antoniopurificato/NLP/semeval2023task3bundle-v2'
os.chdir(PATH)
wandb.login(key = '88987d90526e97d3144c1c3c7ff85ae9b3ea37ad')
torch.manual_seed(21)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
NUM_LABELS = 23

def tuple_to_list(my_tuple):
  my_list = []
  for name in my_tuple:
    my_list.append(name)
  output = ','.join(my_list).replace("'","")
  return output


#Data handling
def set_folder(data_type, language= 'en'):
  if data_type == 'train':
    input_folder = 'data/{}/train-articles-subtask-3/'.format(language)
    labels_folder =  'data/{}/train-labels-subtask-3.txt'.format(language)
  if data_type == 'dev':
    input_folder = 'data/{}/dev-articles-subtask-3/'.format(language)
    labels_folder = None
  return input_folder,labels_folder

def make_dataframe(data_type = 'train', language = 'en'):
    #MAKE TXT DATAFRAME
    input_folder,labels_fn = set_folder(data_type, language)
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD = fil[7:].split('.')[0]
        lines = list(enumerate(open(input_folder+fil,'r',encoding='utf-8').read().splitlines(),1))
        text.extend([(iD,) + line for line in lines])

    df_text = pd.DataFrame(text, columns=['id','line','text'])
    df_text.id = df_text.id.apply(int)
    df_text.line = df_text.line.apply(int)
    df_text = df_text[df_text.text.str.strip().str.len() > 0].copy()
    df_text = df_text.set_index(['id','line'])
    
    df = df_text

    if labels_fn:
        #MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn,sep='\t',encoding='utf-8',header=None)
        labels = labels.rename(columns={0:'id',1:'line',2:'labels'})
        labels = labels.set_index(['id','line'])
        #labels = labels[labels.labels.notna()].copy()
        labels = labels.fillna('')
        #JOIN
        df = labels.join(df_text)[['text','labels']]

    return df.reset_index()

def load_data_tt_split(data_type = 'train', language = 'en'):
    if data_type == 'train':
      train_df = make_dataframe(data_type = 'train', language = language)
    if data_type == 'dev':
      train_df = make_dataframe(data_type = 'dev', language = language)
      
    all_idxs = train_df["id"].to_numpy()
    all_lines = train_df["line"].to_numpy()
    all_data = train_df["text"].to_numpy()
    if data_type == 'train':
      all_labels = my_binarizer_task1.transform(train_df['labels'].fillna('').str.split(',').values)
      return all_idxs, all_lines, all_data, torch.tensor(all_labels)
    if data_type == 'dev':
      return all_idxs, all_lines, all_data#, torch.tensor(all_labels)

my_binarizer_task1 = MultiLabelBinarizer()
classes_file = "scorers/techniques_subtask3.txt"
labels_name1 = []
with open(classes_file, "r") as f:
    for line in f.readlines():
        labels_name1.append(line.rstrip())
#labels_name1.pop(-1)
labels_name1.sort()  # MultiLabelBinarizer sorts the labels
my_binarizer_task1.fit([labels_name1]);


# Dataset class for single models
class PersTecData_tt_split(torch.utils.data.Dataset):
    def __init__(self, data_type="train", tokenizer=None, language = 'en'):
        self.data_type = data_type
        self.language = language
        if self.data_type == 'train':
          self.idxs, self.lines, X, self.y = load_data_tt_split(self.data_type, self.language)
        else:
          self.idxs, self.lines, X = load_data_tt_split(self.data_type, self.language)
        self.tokenized = False
        if tokenizer != None:
            self.tokenized = True
            tokenized = tokenizer(
                X.tolist(), padding="max_length", truncation=True, max_length=128
            )
            self.input_ids = torch.tensor(tokenized["input_ids"])
            self.attention_mask = torch.tensor(tokenized["attention_mask"])
        else:
            self.X = X

    def __getitem__(self, index):
        sample = self.input_ids[index]
        mask = self.attention_mask[index]
        a = torch.squeeze(torch.tensor(self.idxs)[index])
        b = torch.squeeze(torch.tensor(self.lines)[index])
        if self.data_type == 'train':
          label = torch.squeeze(self.y[index])
          return sample, mask, label, a, b
        if self.data_type == 'dev':
          return sample, mask, a, b

    def __len__(self):
        if self.tokenized:
            return self.input_ids.shape[0]
        else:
            return self.X.shape[0]

#Classifier
class PLMClassifier_tt_split(pl.LightningModule):
    def __init__(self, plm):
        super().__init__()
        self.plm = plm
        self.learning_rate = 2e-5
        self.n_warmup_steps = 500
        self.criterion = nn.BCELoss()
        self.thresholds = [x / 10 for x in range(0, 11)]
        self.result = {}

    def forward(self, samples, masks):
        x = self.plm(samples, masks)
        return torch.sigmoid(x.logits)

    def training_step(self, batch, batch_idx):
        batch_ids, batch_mask, labels, _, _ = batch
        preds = self(samples=batch_ids, masks=batch_mask)
        loss = self.criterion(preds, labels.float())
        #wandb.log({"Training loss": loss.item()}) 
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch_ids, batch_mask, article, line = batch
        preds = self(samples=batch_ids, masks=batch_mask)
        for threshold in self.thresholds:
          predictions = torch.greater(preds,
                                      torch.ones(preds.shape).cuda() * threshold)
          for article_number, line_number, output in \
         zip(article.tolist(),line.tolist(),
             my_binarizer_task1.inverse_transform(predictions.cpu())):
            self.result.update({(article_number,line_number,threshold):output})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.n_warmup_steps
        )
        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )



if model_name == 'Bert':
  if LANGUAGE == 'en':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=NUM_LABELS)
  if LANGUAGE == 'it':
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-italian-xxl-cased", num_labels=NUM_LABELS)
  if LANGUAGE == 'ru':
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased", num_labels=NUM_LABELS)
  if LANGUAGE == 'po':
    tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "dkleczek/bert-base-polish-uncased-v1", num_labels=NUM_LABELS)
  if LANGUAGE == 'fr':
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-french-europeana-cased", num_labels=NUM_LABELS)
  if LANGUAGE == 'ge':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-german-cased", num_labels=NUM_LABELS)
if model_name == 'RoBERTa':
  if LANGUAGE == 'en':
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=NUM_LABELS)
  if LANGUAGE == 'po':
    tokenizer = AutoTokenizer.from_pretrained("sdadas/polish-roberta-large-v2", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "sdadas/polish-roberta-large-v2", num_labels=NUM_LABELS)
  if LANGUAGE == 'ru':
    tokenizer = AutoTokenizer.from_pretrained("blinoff/roberta-base-russian-v0", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "blinoff/roberta-base-russian-v0", num_labels=NUM_LABELS)
    
if model_name == 'XLNet':
  if LANGUAGE == 'en':
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "xlnet-base-cased", num_labels=NUM_LABELS)
if model_name == 'DeBERTa':
  if LANGUAGE == 'en':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-base", num_labels=NUM_LABELS)
if model_name == 'alBERT':
  if LANGUAGE == 'en':
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", use_fast=True)
    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "albert-base-v2", num_labels=NUM_LABELS)

#Dasaset and Dataloader creation 
dataset_train_tt_split = PersTecData_tt_split(data_type="train",tokenizer=tokenizer, language = LANGUAGE)
train_loader_tt_split = DataLoader(dataset_train_tt_split, batch_size=8,
                                   num_workers=0, pin_memory=True)
 
dataset_val_tt_split = PersTecData_tt_split(data_type="dev",
                                            tokenizer=tokenizer, language = LANGUAGE)
val_loader_tt_split = DataLoader(dataset_val_tt_split, batch_size=8,
                                 num_workers=0, pin_memory=True)

#Data visualization
print(dataset_train_tt_split.input_ids[0])
print(tokenizer.convert_ids_to_tokens(dataset_train_tt_split.input_ids[0]))
label = dataset_train_tt_split.y[0]
print(label)
print(my_binarizer_task1.inverse_transform(label.reshape(1, -1)))

#new_model = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/Bert_10-v1.ckpt', plm = classification_model)

#Training
model = PLMClassifier_tt_split(classification_model)

#Log of the training
run_name = model_name + '_' + LANGUAGE + '_' + str(EPOCHS)
logger = WandbLogger()
checkpoint_callback = ModelCheckpoint(
    dirpath='../lightning_logs',
    filename= run_name)

wandb.init(
      # Set the project where this run will be logged
      name = run_name,
      project="NUANS", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10) 
      # Track hyperparameters and run metadata
      config={
      "epochs": EPOCHS
      })


trainer1 = pl.Trainer(gpus = 1,max_epochs=EPOCHS, logger = logger, callbacks=[checkpoint_callback])

trainer1.fit(model, train_loader_tt_split, 
             val_loader_tt_split)
wandb.finish()

#Validation
def test_classifier(model, data_loader, thresholds):
    model.cuda()
    model.eval()
    result = {}
    for i, batch in enumerate(data_loader):
      batch_ids, batch_mask, article, line = batch
      preds = model(batch_ids.cuda(), batch_mask.cuda())
      for threshold in thresholds:
        predictions = torch.greater(preds,
                                      torch.ones(preds.shape).cuda() * threshold)
        for article_number, line_number, output in \
         zip(article.tolist(),line.tolist(),
             my_binarizer_task1.inverse_transform(predictions.cpu())):
          result.update({(article_number,line_number,threshold):output})
    return result

thresholds = [x / 10 for x in range(0, 11)]
result = test_classifier(model, val_loader_tt_split, thresholds)

with open('../lightning_logs/{}_dictionary.pkl'.format(run_name), 'wb') as f:
    pickle.dump(result, f)

my_list = []
for b,c in zip(result.keys(),result.values()):
  if b[2] == THRESHOLD:
    my_list.append([b[0],b[1],tuple_to_list(c)])

my_df = pd.DataFrame(my_list, columns= ['Article_id','Line_id','Techniques'])

my_df.to_csv(run_name + '_output.txt',header=None, index=None, sep='\t')
""" #Test

for a,b in zip(result.keys(),result.values()):
  if a[2] == 0.7:
    print(a[0],a[1],b)

my_list = []
for b,c in zip(result.keys(),result.values()):
  if b[2] == 0.6:
    my_list.append([b[0],b[1],tuple_to_list(c)])

my_df = pd.DataFrame(my_list, columns= ['Article_id','Line_id','Techniques'])

my_df.to_csv(run_name + '_output.txt',header=None, index=None, sep='\t')

#with open('../lightning_logs/{}_dictionary.pkl'.format(run_name), 'wb') as f:
#    pickle.dump(result, f)

#with open('../lightning_logs/{}_dictionary.pkl'.format(run_name), 'rb') as f:
    #loaded_dict = pickle.load(f) """

