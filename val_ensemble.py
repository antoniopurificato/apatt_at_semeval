import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_constant_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import argparse
import os
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader


PATH = '/home/antpur/projects/apatt_at_semeval/semeval2023task3bundle-v3'
os.chdir(PATH)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--language', type=str, help='Language')
parser.add_argument('--threshold', type=float, help='Value of the threshold')
parser.add_argument('--mode', type=str, help='online or offline')
args = parser.parse_args()
EPOCHS = args.epochs
LANGUAGE = args.language
THRESHOLD = args.threshold
os.environ['WANDB_MODE'] = args.mode
NUM_LABELS = 23
torch.manual_seed(21)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
run_name = LANGUAGE + '_ensemble'

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
    labels_folder = 'data/{}/dev-labels-subtask-3.txt'.format(language)
  if data_type == 'test':
    input_folder = 'data/{}/test-articles-subtask-3/'.format(language)
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
    if data_type == 'test':
      train_df = make_dataframe(data_type = 'test', language = language)
      
    all_idxs = train_df["id"].to_numpy()
    all_lines = train_df["line"].to_numpy()
    all_data = train_df["text"].to_numpy()
    if data_type == 'train' or data_type == 'dev':
      all_labels = my_binarizer_task1.transform(train_df['labels'].fillna('').str.split(',').values)
      return all_idxs, all_lines, all_data, torch.tensor(all_labels)
    if data_type == 'test':
      return all_idxs, all_lines, all_data#, torch.tensor(all_labels)


class PersTecData_tt_split(torch.utils.data.Dataset):
    def __init__(self, data_type="train", tokenizer=None, language = 'en'):
        self.data_type = data_type
        self.language = language
        if self.data_type == 'train' or self.data_type == 'dev':
          self.idxs, self.lines, X, self.y = load_data_tt_split(self.data_type, self.language)
        if self.data_type == 'test':
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
        if self.data_type == 'train' or self.data_type == 'dev':
          label = torch.squeeze(self.y[index])
          return sample, mask, label, a, b
        if self.data_type == 'test':
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
        batch_ids, batch_mask, labels, _, _ = batch
        preds = self(samples=batch_ids, masks=batch_mask)
        loss = self.criterion(preds, labels.float())
        #wandb.log({"Training loss": loss.item()}) 
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
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

my_binarizer_task1 = MultiLabelBinarizer()
classes_file = "scorers/techniques_subtask3.txt"
labels_name1 = []
with open(classes_file, "r") as f:
    for line in f.readlines():
        labels_name1.append(line.rstrip())
#labels_name1.pop(-1)
labels_name1.sort()  # MultiLabelBinarizer sorts the labels
my_binarizer_task1.fit([labels_name1]);


# Dataset class for ensembles of models
class PersTecDataEnsemble(torch.utils.data.Dataset):
    def __init__(self, data_type="train", tokenizers=None):
        self.data_type = data_type
        if self.data_type == 'train' or self.data_type == 'dev':
          self.idxs, self.lines, self.X, self.y = load_data_tt_split(self.data_type, language = LANGUAGE)
        if self.data_type == 'test':
          self.idxs, self.lines, self.X = load_data_tt_split(self.data_type, language = LANGUAGE)
        self.num_tok = len(tokenizers)
        self.input_ids = []
        self.attention_mask = []
        for tokenizer in tokenizers:
            tokenized = tokenizer(
                self.X.tolist(), padding="max_length", truncation=True, max_length=128
            )
            self.input_ids.append(torch.tensor(tokenized["input_ids"]))
            self.attention_mask.append(torch.tensor(tokenized["attention_mask"]))

    def __getitem__(self, index):
        elem = []
        a = torch.squeeze(torch.tensor(self.idxs)[index])
        b = torch.squeeze(torch.tensor(self.lines)[index])
        for i in range(self.num_tok):
            sample = self.input_ids[i][index]
            mask = self.attention_mask[i][index]
            elem.append([sample, mask])
        return elem, a, b

    def __len__(self):
        return self.X.shape[0]


class EnsembleClassifier(pl.LightningModule):
    def __init__(self, models):
        super().__init__()
        self.models = []
        for model in models:
            self.models.append(model)
        self.n_models = len(self.models)

    def forward(self, batch):
        preds = []
        for i, model in enumerate(self.models):
            samples, masks = batch[i]
            device = model.device
            samples = samples.to(device)
            masks = masks.to(device)
            x = model(samples, masks)
            preds.append(x)
        preds = torch.stack(preds)
        pred = torch.mean(preds, axis=0)
        return pred



if LANGUAGE == 'en':
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    dataset_train_bert = PersTecData_tt_split(data_type="train",tokenizer=tokenizer_bert, language = LANGUAGE)
    train_loader_bert = DataLoader(dataset_train_bert, batch_size=8,num_workers=0, pin_memory=True)
    dataset_val_bert = PersTecData_tt_split(data_type="dev",tokenizer=tokenizer_bert, language = LANGUAGE)
    val_loader_tt_split = DataLoader(dataset_val_bert, batch_size=8,num_workers=0, pin_memory=True)
    
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    dataset_train_RoBERTa = PersTecData_tt_split(data_type="train", tokenizer=tokenizer_roberta, language=LANGUAGE)
    train_loader_RoBERTa = DataLoader(dataset_train_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)
    dataset_val_RoBERTa = PersTecData_tt_split(data_type="dev", tokenizer=tokenizer_roberta, language=LANGUAGE)
    val_loader_RoBERTa = DataLoader(dataset_val_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)

    tokenizer_xlnet = AutoTokenizer.from_pretrained("xlnet-base-cased", use_fast=True)
    dataset_train_xlnet = PersTecData_tt_split(data_type="train", tokenizer=tokenizer_xlnet, language=LANGUAGE)
    train_loader_xlnet = DataLoader(dataset_train_xlnet, batch_size=8, num_workers=2, pin_memory=True)
    dataset_val_xlnet = PersTecData_tt_split(data_type="dev", tokenizer=tokenizer_xlnet, language=LANGUAGE)
    val_loader_xlnet = DataLoader(dataset_val_xlnet, batch_size=8, num_workers=2, pin_memory=True)

    tokenizer_deberta = AutoTokenizer.from_pretrained("microsoft/deberta-base", use_fast=True)
    dataset_train_deberta = PersTecData_tt_split(data_type="train", tokenizer=tokenizer_deberta, language=LANGUAGE)
    train_loader_deberta = DataLoader(dataset_train_deberta, batch_size=8, num_workers=2, pin_memory=True)
    dataset_val_deberta = PersTecData_tt_split(data_type="dev", tokenizer=tokenizer_deberta, language=LANGUAGE)
    val_loader_deberta = DataLoader(dataset_val_deberta, batch_size=8, num_workers=2, pin_memory=True)

    tokenizer_albert = AutoTokenizer.from_pretrained("albert-base-v2", use_fast=True)
    dataset_train_albert = PersTecData_tt_split(data_type="train", tokenizer=tokenizer_albert, language=LANGUAGE)
    train_loader_albert = DataLoader(dataset_train_albert, batch_size=8, num_workers=2, pin_memory=True)
    dataset_val_albert = PersTecData_tt_split(data_type="dev", tokenizer=tokenizer_albert, language=LANGUAGE)
    val_loader_albert = DataLoader(dataset_val_albert, batch_size=8, num_workers=2, pin_memory=True)
    
    dataset_val1_ensemble = PersTecDataEnsemble(data_type="dev",
        tokenizers=[tokenizer_bert, tokenizer_roberta, tokenizer_albert, tokenizer_deberta, tokenizer_xlnet])
    val_loader_ensemble = DataLoader(dataset_val1_ensemble, batch_size=8, num_workers=2, pin_memory=True)

    bert = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=NUM_LABELS)
    roberta = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=NUM_LABELS)
    xlnet = AutoModelForSequenceClassification.from_pretrained(
    "xlnet-base-cased", num_labels=NUM_LABELS)
    deberta = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-base", num_labels=NUM_LABELS)
    albert = AutoModelForSequenceClassification.from_pretrained(
    "albert-base-v2", num_labels=NUM_LABELS)

    model_bert = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/Bert_{}_10-v1.ckpt'.format(LANGUAGE), plm = bert).to(device)
    model_roberta = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/RoBERTa_{}_10-v1.ckpt'.format(LANGUAGE), plm = roberta).to(device)
    model_xlnet = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/XLNet_{}_10-v1.ckpt'.format(LANGUAGE), plm = xlnet).to(device)
    model_deberta = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/DeBERTa_{}_10-v1.ckpt'.format(LANGUAGE), plm = deberta).to(device)
    model_albert = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/_{}_10-v1.ckpt'.format(LANGUAGE), plm = albert).to(device)
    ensemble = EnsembleClassifier(
    [model_bert, model_roberta, model_xlnet, model_deberta, model_albert])

if LANGUAGE == 'it':
    tokenizer_bert = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased", use_fast=True)
    dataset_train_bert = PersTecData_tt_split(data_type="train",tokenizer=tokenizer_bert, language = LANGUAGE)
    train_loader_bert = DataLoader(dataset_train_bert, batch_size=8,num_workers=0, pin_memory=True)
    dataset_val_bert = PersTecData_tt_split(data_type="dev",tokenizer=tokenizer_bert, language = LANGUAGE)
    val_loader_tt_split = DataLoader(dataset_val_bert, batch_size=8,num_workers=0, pin_memory=True)

    dataset_val1_ensemble = PersTecDataEnsemble(data_type="dev",tokenizers=[tokenizer_bert])
    val_loader_ensemble = DataLoader(dataset_val1_ensemble, batch_size=8, num_workers=2, pin_memory=True)

    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-italian-xxl-cased", num_labels=NUM_LABELS)
    model = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/Bert_{}_10-v1.ckpt'.format(LANGUAGE), plm = classification_model).to(device)
    ensemble = EnsembleClassifier([model])
    
if LANGUAGE == 'ru':
    tokenizer_bert = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", use_fast=True)
    dataset_train_bert = PersTecData_tt_split(data_type="train",tokenizer=tokenizer_bert, language = LANGUAGE)
    train_loader_bert = DataLoader(dataset_train_bert, batch_size=8,num_workers=0, pin_memory=True)
    dataset_val_bert = PersTecData_tt_split(data_type="dev",tokenizer=tokenizer_bert, language = LANGUAGE)
    val_loader_tt_split = DataLoader(dataset_val_bert, batch_size=8,num_workers=0, pin_memory=True)
    #dataset_test_bert = PersTecData_tt_split(data_type="test",tokenizer=tokenizer_bert, language = LANGUAGE)
    #test_loader_tt_split = DataLoader(dataset_test_bert, batch_size=8,num_workers=0, pin_memory=True)
    
    tokenizer_roberta = AutoTokenizer.from_pretrained("blinoff/roberta-base-russian-v0", use_fast=True)
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    dataset_train_RoBERTa = PersTecData_tt_split(data_type="train", tokenizer=tokenizer_roberta, language=LANGUAGE)
    train_loader_RoBERTa = DataLoader(dataset_train_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)
    dataset_val_RoBERTa = PersTecData_tt_split(data_type="dev", tokenizer=tokenizer_roberta, language=LANGUAGE)
    val_loader_RoBERTa = DataLoader(dataset_val_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)
    #dataset_test_RoBERTa = PersTecData_tt_split(data_type="test", tokenizer=tokenizer_roberta, language=LANGUAGE)
    #test_loader_RoBERTa = DataLoader(dataset_test_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)


    dataset_val1_ensemble = PersTecDataEnsemble(data_type="dev",tokenizers=[tokenizer_bert, tokenizer_roberta])
    val_loader_ensemble = DataLoader(dataset_val1_ensemble, batch_size=8, num_workers=2, pin_memory=True)
    
    bert = AutoModelForSequenceClassification.from_pretrained(
        "DeepPavlov/rubert-base-cased", num_labels=NUM_LABELS)
    roberta = AutoModelForSequenceClassification.from_pretrained(
    "blinoff/roberta-base-russian-v0", num_labels=NUM_LABELS)

    model_bert = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/Bert_{}_10-v1.ckpt'.format(LANGUAGE), plm = bert).to(device)
    model_roberta = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/RoBERTa_{}_10-v1.ckpt'.format(LANGUAGE), plm = roberta).to(device)
    ensemble = EnsembleClassifier([model_bert, model_roberta])

if LANGUAGE == 'po':
    tokenizer_bert = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1", use_fast=True)
    dataset_train_bert = PersTecData_tt_split(data_type="train",tokenizer=tokenizer_bert, language = LANGUAGE)
    train_loader_bert = DataLoader(dataset_train_bert, batch_size=8,num_workers=0, pin_memory=True)
    dataset_val_bert = PersTecData_tt_split(data_type="dev",tokenizer=tokenizer_bert, language = LANGUAGE)
    val_loader_tt_split = DataLoader(dataset_val_bert, batch_size=8,num_workers=0, pin_memory=True)
    dataset_test_bert = PersTecData_tt_split(data_type="test",tokenizer=tokenizer_bert, language = LANGUAGE)
    test_loader_tt_split = DataLoader(dataset_test_bert, batch_size=8,num_workers=0, pin_memory=True)
    
    tokenizer_roberta = AutoTokenizer.from_pretrained("sdadas/polish-roberta-large-v2", use_fast=True)
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    dataset_train_RoBERTa = PersTecData_tt_split(data_type="train", tokenizer=tokenizer_roberta, language=LANGUAGE)
    train_loader_RoBERTa = DataLoader(dataset_train_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)
    dataset_val_RoBERTa = PersTecData_tt_split(data_type="dev", tokenizer=tokenizer_roberta, language=LANGUAGE)
    val_loader_RoBERTa = DataLoader(dataset_val_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)
    dataset_test_RoBERTa = PersTecData_tt_split(data_type="test", tokenizer=tokenizer_roberta, language=LANGUAGE)
    test_loader_RoBERTa = DataLoader(dataset_test_RoBERTa, batch_size=8, num_workers=2, pin_memory=True)

    dataset_val1_ensemble = PersTecDataEnsemble(data_type="dev",tokenizers=[tokenizer_bert, tokenizer_roberta])
    val_loader_ensemble = DataLoader(dataset_val1_ensemble, batch_size=8, num_workers=2, pin_memory=True)
    
    bert = AutoModelForSequenceClassification.from_pretrained(
        "dkleczek/bert-base-polish-uncased-v1", num_labels=NUM_LABELS)
    roberta = AutoModelForSequenceClassification.from_pretrained(
        "sdadas/polish-roberta-large-v2", num_labels=NUM_LABELS)

    model_bert = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/Bert_{}_10-v1.ckpt'.format(LANGUAGE), plm = bert).to(device)
    model_roberta = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/RoBERTa_{}_10-v1.ckpt'.format(LANGUAGE), plm = roberta).to(device)
    ensemble = EnsembleClassifier([model_bert, model_roberta])

if LANGUAGE == 'fr':
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased", use_fast=True)

    dataset_val1_ensemble = PersTecDataEnsemble(data_type="dev",tokenizers=[tokenizer_bert])
    val_loader_ensemble = DataLoader(dataset_val1_ensemble, batch_size=8, num_workers=2, pin_memory=True)

    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-french-europeana-cased", num_labels=NUM_LABELS)
    model = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/Bert_{}_10-v1.ckpt'.format(LANGUAGE), plm = classification_model).to(device)
    ensemble = EnsembleClassifier([model])

if LANGUAGE == 'ge':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased", use_fast=True)

    dataset_val1_ensemble = PersTecDataEnsemble(data_type="dev",tokenizers=[tokenizer_bert])
    val_loader_ensemble = DataLoader(dataset_val1_ensemble, batch_size=8, num_workers=2, pin_memory=True)

    classification_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-german-cased", num_labels=NUM_LABELS)
    model = PLMClassifier_tt_split.load_from_checkpoint('../lightning_logs/Bert_{}_10-v1.ckpt'.format(LANGUAGE), plm = classification_model).to(device)
    ensemble = EnsembleClassifier([model])


def test_classifier_ensemble(model, data_loader, thresholds):
    model.cuda()
    model.eval()
    true_labels = []
    result = {}
    for i, batch in tqdm(enumerate(data_loader)):
      batch_ids, article, line = batch
      preds = model(batch_ids).detach()
      for threshold in thresholds:
        predictions = torch.greater(preds.cuda(),
                                      torch.ones(preds.shape).cuda() * threshold)
        for article_number, line_number, output in \
         zip(article.tolist(),line.tolist(),
             my_binarizer_task1.inverse_transform(predictions.cpu())):
          result.update({(article_number,line_number,threshold):output})
    return result

thresholds = [x / 10 for x in range(0, 11)]
result_ensemble = test_classifier_ensemble(ensemble, val_loader_ensemble, thresholds)

my_list = []
for b,c in zip(result_ensemble.keys(),result_ensemble.values()):
  if b[2] == THRESHOLD:
    my_list.append([b[0],b[1],tuple_to_list(c)])

with open('../lightning_logs/{}_dictionary.pkl'.format(run_name), 'wb') as f:
    pickle.dump(result_ensemble, f)

my_df = pd.DataFrame(my_list, columns= ['Article_id','Line_id','Techniques'])

my_df.to_csv('../lightning_logs/' + run_name + '_output.txt',header=None, index=None, sep='\t')