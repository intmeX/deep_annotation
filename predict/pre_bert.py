import numpy as np
import time
import torch
import random
from transformers import BertForSequenceClassification, \
                        BertTokenizerFast, BertConfig, AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from scripts import XML
from learn import config


data_path = '../data/base/problems_with_tag50.xml'
max_length = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 20
batch_size = 20
vocab_len = 3000000
vec_dim = 300
hidden_dim = 20
num_layers = 3
num_classes = 50
model_name = 'bert-base-uncased'
hidden_dropout_prob = 0.3
weight_decay = 0.01

# BERT参数设置
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
config.num_labels = num_classes
config.hidden_dropout_prob = hidden_dropout_prob


# 定义分词器
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')


# 自定义Dataset子类
class CpcText(Dataset):
    def __init__(self, data):
        # self.dataset = pd.read_csv(path_to_file, sep="\t", names=["Phrase", "Sentiment"])
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        desc = self.dataset[idx]["desc"]
        label = self.dataset[idx]["label"]
        sample = {"desc": desc, "label": label}
        return sample


def data_trans(data):
    global max_length
    global num_classes
    global device
    res = dict()
    res['desc'] = data['stmt']
    label = [int(j) for j in data['tag'].split(',')]
    oh = [0] * num_classes
    for j in label:
        oh[j] = 1
    res['label'] = torch.tensor(oh, dtype=torch.float32).to(device)
    return res


def data_prepare():
    global data_path
    global batch_size
    data_xml = XML.read_xml(data_path)['problem']
    data = []
    for i in data_xml:
        data.append(data_trans(i))
    random.seed(time.time())
    random.shuffle(data)
    data_train, data_validate = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    train_loader = DataLoader(CpcText(data_train), batch_size=batch_size, num_workers=0)
    validate_loader = DataLoader(CpcText(data_validate), batch_size=batch_size, num_workers=0)
    return train_loader, validate_loader


def training(model, dataloader, optimizer, criterion, thr=0.8):
    global tokenizer
    global num_classes
    global device
    global max_length
    model.train()
    epoch_loss = 0
    epoch_acc_tp = 0
    epoch_acc_tr = 0
    for i, batch in enumerate(dataloader):
        desc = batch["desc"]
        # label = torch.tensor(batch["label"], dtype=torch.long).to(device)
        label = batch["label"]

        tokenizer_res = tokenizer(
            desc,
            max_length=max_length,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            verbose=True,
            return_tensors="pt")
        tokenizer_res = tokenizer_res.to(device)

        optimizer.zero_grad()

        output = model(**tokenizer_res, labels=label)
        prob = output[1]

        loss = criterion(prob.view(-1, num_classes), label)

        p = torch.sum(prob > thr, axis=1) + 1
        r = torch.sum(label > thr, axis=1) + 1
        t = torch.sum((prob > thr) & (label > thr), axis=1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc_tp += torch.mean(t / p).item()
        epoch_acc_tr += torch.mean(t / r).item()

        if i % 30 == 29:
            print("{:>5} loss: {}\ttp: {}\ttr: {}".format(i, epoch_loss / (i + 1),
                                                          epoch_acc_tp / (i + 1),
                                                          epoch_acc_tr / (i + 1)))
    return epoch_loss / len(dataloader), epoch_acc_tp / len(dataloader), epoch_acc_tr / len(dataloader)


def evaluting(model, dataloader, criterion, thr=0.7):
    global tokenizer
    global num_classes
    global device
    model.eval()
    epoch_loss = 0
    epoch_acc_tp = 0
    epoch_acc_tr = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            desc = batch["desc"]
            # label = torch.tensor(batch["label"], dtype=torch.long).to(device)
            label = batch["label"]

            tokenizer_res = tokenizer(
                desc,
                max_length=max_length,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                verbose=True,
                return_tensors="pt")
            tokenizer_res = tokenizer_res.to(device)

            output = model(**tokenizer_res, labels=label)
            prob = output[1]

            p = torch.sum(prob > thr, axis=1) + 1
            r = torch.sum(label > thr, axis=1) + 1
            t = torch.sum((prob > thr) & (label > thr), axis=1)

            loss = criterion(prob.view(-1, num_classes), label)

            epoch_loss += loss.item()
            epoch_acc_tp += torch.mean(t / p).item()
            epoch_acc_tr += torch.mean(t / r).item()

            if i % 30 == 29:
                print("{:>5} loss: {}\ttp: {}\ttr: {}".format(i, epoch_loss / (i + 1),
                                                              epoch_acc_tp / (i + 1),
                                                              epoch_acc_tr / (i + 1)))

    return epoch_loss / len(dataloader), epoch_acc_tp / len(dataloader), epoch_acc_tr / len(dataloader)


if __name__ == '__main__':
    train_loader, validate_loader = data_prepare()
    print(train_loader.dataset[2])
    print(validate_loader.dataset[2])
    # 模型加载
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=config)
    model.to(device)
    # model = lstm_ml.LSTM(None, vocab_len, vec_dim, hidden_dim, num_layers, num_classes=num_classes).to(device)
    # optimizer = Adam(lr=1e-4, eps=1e-8, weight_decay=0.01)
    # 一是去掉无用的设置 而是构造字典列表以使AdamW可以接受该参数
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    loss_func = CrossEntropyLoss()

    times = []
    last_epoch = 0

    for i in range(epoch):
        torch.cuda.synchronize()
        start = time.time()
        # try:
        train_loss, train_accp, train_accr = training(model, train_loader, optimizer, loss_func)
        print("\ntraining epoch {:<1} loss: {}\taccp: {}\taccr: {}\n".format(i + 1, train_loss, train_accp, train_accr))
        eval_loss, eval_accp, eval_accr = evaluting(model, validate_loader, loss_func)
        print("\nevaluting epoch {:<1} loss: {}\taccp: {}\taccr: {}\n".format(i + 1, eval_loss, eval_accp, eval_accr))
        # except:
        #     torch.save(model, configs.model_data_path + 'cpc_lstm.model')
        #     break

        last_epoch = i + 1
        torch.cuda.synchronize()
        times.append(time.time() - start)

    if last_epoch == epoch:
        torch.save(model, config.model_data_path + 'cpc_bert.model')
    print("\ntimecost:", times, "\n")

