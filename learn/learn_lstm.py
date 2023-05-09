import numpy as np
import time
import torch
import random
from transformers import BertTokenizerFast
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from gensim.models import KeyedVectors
from scripts import XML
from model import lstm_ml
from configs.config import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data_path = '../data/base/problems_with_tag50.xml'
# max_length = 200
# epoch = 100
# batch_size = 20
# vocab_len = 3000000
# vec_dim = 300
# hidden_dim = 20
# num_layers = 3
# num_classes = 50


# 定义分词器
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
embedding = KeyedVectors.load_word2vec_format(model_data_path + 'GoogleNews-vectors-negative300.bin', binary=True)
# embedding = torch.load(config.model_data_path + 'GoogleNews-vectors-negative300.model').to(device)


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
    global tokenizer
    global embedding
    global vec_dim
    global num_classes
    global device
    pad = np.random.normal(loc=0, scale=1, size=(vec_dim,))
    res = dict()
    res['desc'] = tokenizer.tokenize(data['stmt'])
    length = len(res['desc'])
    if length > max_length:
        res['desc'] = res['desc'][:max_length]
        length = max_length
    i = 0
    while i < length:
        try:
            res['desc'][i] = embedding[res['desc'][i]]
        except:
            res['desc'][i] = pad
        i += 1
    if length < max_length:
        res['desc'] += [pad] * (max_length - length)
    res['desc'] = torch.tensor(np.array(res['desc']), dtype=torch.float32).to(device)
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
    model.train()
    epoch_loss = 0
    epoch_acc_tp = 0
    epoch_acc_tr = 0
    for i, batch in enumerate(dataloader):
        desc = batch["desc"]
        # label = torch.tensor(batch["label"], dtype=torch.long).to(device)
        label = batch["label"]

        optimizer.zero_grad()

        prob = model(desc)
        p = torch.sum(prob > thr, axis=1) + 1
        r = torch.sum(label > thr, axis=1) + 1
        t = torch.sum((prob > thr) & (label > thr), axis=1)

        # loss = criterion(prob.view(-1, num_classes), label.view(-1))
        loss = criterion(prob.view(-1, num_classes), label)

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

            prob = model(desc)
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


def main():
    train_loader, validate_loader = data_prepare()
    print(train_loader.dataset[2])
    print(validate_loader.dataset[2])
    model = lstm_ml.LSTM(None, vocab_len, vec_dim, hidden_dim, num_layers, num_classes=num_classes).to(device)
    # optimizer = Adam(lr=1e-4, eps=1e-8, weight_decay=0.01)
    # 参考博客 一是去掉无用的设置 而是构造字典列表以使AdamW可以接受该参数
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_func = CrossEntropyLoss()

    times = []
    model_path = './model/'
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
        #     torch.save(model, config.model_data_path + 'cpc_lstm.model')
        #     break

        last_epoch = i + 1
        torch.cuda.synchronize()
        times.append(time.time() - start)

    if last_epoch == epoch:
        torch.save(model, model_data_path + 'cpc_lstm.model')
    print("\ntimecost:", times, "\n")


if __name__ == '__main__':
    main()

