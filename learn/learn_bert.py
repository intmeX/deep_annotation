import os
import numpy as np
import pandas as pd
import time
import torch
import random
from transformers import BertForSequenceClassification, \
                        BertTokenizerFast, BertConfig, AdamW, \
                        get_cosine_schedule_with_warmup, \
                        get_linear_schedule_with_warmup, \
                        get_constant_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from scripts import XML
from configs.config import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
last_epoch = 0


# 自定义Dataset子类
class CpcText(Dataset):
    def __init__(self, data):
        # self.dataset = pd.read_csv(path_to_file, sep="\t", names=["Phrase", "Sentiment"])
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        desc = self.dataset[idx][feature]
        tag = self.dataset[idx][label]
        sample = {feature: desc, label: tag}
        return sample


def data_trans(data):
    global max_length
    global num_classes
    global device
    res = dict()
    res[feature] = data[feature]
    tag = [int(j) for j in data[label].split(',')]
    oh = [0] * num_classes
    for j in tag:
        oh[j] = 1
    res[label] = torch.tensor(oh, dtype=torch.float32)
    return res


def get_data(path):
    global data_root
    if path[-3:] == 'xml':
        res = XML.read_xml(data_root + path)[xml_name]
    elif path[-3:] == 'csv':
        dic = pd.read_csv(data_root + path).to_dict('list')
        res = []
        length = 0
        for i in dic:
            length = len(dic[i])
            break
        for j in range(length):
            item = dict()
            for i in dic:
                item[i] = dic[i][j]
            res.append(item)
    else:
        raise Exception('invalid data file format')
    return res


def data_prepare():
    global data_path
    global batch_size
    data_raw = get_data(data_path)
    data = []
    for i in data_raw:
        data.append(data_trans(i))
    random.seed(1)
    random.shuffle(data)
    data_train, data_validate = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    train_loader = DataLoader(CpcText(data_train), batch_size=batch_size, num_workers=0)
    validate_loader = DataLoader(CpcText(data_validate), batch_size=batch_size, num_workers=0)
    return train_loader, validate_loader


def metric_calc(tag, pred):
    return (
        # metrics.accuracy_score(tag, pred),
        metrics.precision_score(tag, pred, average='samples'),
        metrics.recall_score(tag, pred, average='samples'),
        metrics.hamming_loss(tag, pred),
        metrics.f1_score(tag, pred, average='micro'),
    )


def training(model, dataloader, optimizer, scheduler, criterion, tokenizer, writer, thr=threshold):
    global num_classes
    global device
    global max_length
    global last_epoch
    global warmup
    global decay_start
    model.train()
    epoch_loss = 0
    epoch_acc_prec = 0
    epoch_acc_recall = 0
    epoch_acc_haming = 0
    batch_num = last_epoch * len(dataloader)
    for i, batch in enumerate(dataloader):
        desc = batch[feature]
        tag = batch[label].to(device)

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

        output = model(**tokenizer_res, labels=tag)
        prob = output[1]

        loss = criterion(prob.view(-1, num_classes), tag)

        pred = prob > thr

        loss.backward()
        optimizer.step()
        if batch_num + i < warmup or batch_num + i > decay_start:
            scheduler.step()

        (
            # train_acc,
            train_prec,
            train_recall,
            train_haming,
            train_micro_f1,
        ) = metric_calc(tag.cpu().numpy(), pred.cpu().numpy())
        epoch_loss += loss.item()
        epoch_acc_prec += train_prec
        epoch_acc_recall += train_recall
        epoch_acc_haming += train_haming

        if i % 20 == 19:
            lr = scheduler.get_lr()[0]
            writer.add_scalar('loss/train', loss.item(), batch_num + i)
            writer.add_scalar("prec/train", train_prec, batch_num + i)
            writer.add_scalar("recall/train", train_recall, batch_num + i)
            writer.add_scalar("hamming/train", train_haming, batch_num + i)
            writer.add_scalar("micro_f1/train", train_micro_f1, batch_num + i)
            writer.add_scalar("learning_rate", lr, batch_num + i)

            print("{:>5} loss: {}\tprec: {}\trecall: {}\thaming: {}\tlr: {}".format(
                i,
                epoch_loss / (i + 1),
                epoch_acc_prec / (i + 1),
                epoch_acc_recall / (i + 1),
                epoch_acc_haming / (i + 1),
                scheduler.get_lr()[0],
            ))
    return epoch_loss / len(dataloader), epoch_acc_prec / len(dataloader), epoch_acc_recall / len(dataloader), epoch_acc_haming / len(dataloader)


def evaluting(model, dataloader, criterion, tokenizer, writer, thr=threshold):
    global num_classes
    global device
    global last_epoch
    model.eval()
    epoch_loss = 0
    epoch_acc_prec = 0
    epoch_acc_recall = 0
    epoch_acc_haming = 0
    batch_num = last_epoch * len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            desc = batch[feature]
            # label = torch.tensor(batch[label], dtype=torch.long).to(device)
            tag = batch[label].to(device)

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

            output = model(**tokenizer_res, labels=tag)
            prob = output[1]

            pred = prob > thr

            loss = criterion(prob.view(-1, num_classes), tag)

            (
                # val_acc,
                val_prec,
                val_recall,
                val_haming,
                val_micro_f1,
            ) = metric_calc(tag.cpu().numpy(), pred.cpu().numpy())
            epoch_loss += loss.item()
            epoch_acc_prec += val_prec
            epoch_acc_recall += val_recall
            epoch_acc_haming += val_haming

            if i % 20 == 19:
                writer.add_scalar('loss/val', loss.item(), batch_num + i)
                writer.add_scalar("prec/val", val_prec, batch_num + i)
                writer.add_scalar("recall/val", val_recall, batch_num + i)
                writer.add_scalar("hamming/val", val_haming, batch_num + i)
                writer.add_scalar("micro_f1/val", val_micro_f1, batch_num + i)
                print("{:>5} loss: {}\tprec: {}\trecall: {}\thaming: {}".format(
                    i,
                    epoch_loss / (i + 1),
                    epoch_acc_prec / (i + 1),
                    epoch_acc_recall / (i + 1),
                    epoch_acc_haming / (i + 1),
                ))
    return epoch_loss / len(dataloader), epoch_acc_prec / len(dataloader), epoch_acc_recall / len(dataloader), epoch_acc_haming / len(dataloader)


def main():
    global last_epoch
    global device
    global end
    global warmup
    global decay_start
    end = model_path
    if model_path == '_':
        end = time.strftime("%Y%m%d%H%M%S", time.localtime()) + '_bert.model'
    if os.path.exists(metrics_path + end):
        os.mkdir(metrics_path + end)
    writer = SummaryWriter(metrics_path + end)

    # 定义分词器
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')

    train_loader, validate_loader = data_prepare()
    print(train_loader.dataset[2])
    print(validate_loader.dataset[2])
    # BERT参数设置
    bert_config = BertConfig.from_pretrained(bert_name)
    bert_config.output_hidden_states = False
    bert_config.num_labels = num_classes
    bert_config.hidden_dropout_prob = hidden_dropout_prob
    # 模型加载
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=bert_name, config=bert_config)
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # 从decay_start开始衰减
    train_iters = len(train_loader) * epoch
    if max_iters > 0:
        train_iters = max_iters
    train_iters += warmup - decay_start
    scheduler = None
    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=train_iters)
    elif schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=train_iters)
    elif schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    elif schedule == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.9998)
    else:
        raise Exception('No such scheduler: {}'.format(schedule))
    loss_func = BCEWithLogitsLoss()

    times = []

    for i in range(epoch):
        torch.cuda.synchronize()
        start = time.time()
        # try:
        train_loss, train_acc_prec, train_acc_recall, train_acc_haming = training(model, train_loader, optimizer, scheduler, loss_func, tokenizer, writer)
        print("\ntraining epoch {:<1} loss: {}\tacc_prec: {}\tacc_recall: {}\tacc_haming: {}\n".format(i + 1, train_loss, train_acc_prec, train_acc_recall, train_acc_haming))
        eval_loss, eval_acc_prec, eval_acc_recall, eval_acc_haming = evaluting(model, validate_loader, loss_func, tokenizer, writer)
        print("\nevaluting epoch {:<1} loss: {}\tacc_prec: {}\tacc_recall: {}\tacc_haming: {}\n".format(i + 1, eval_loss, eval_acc_prec, eval_acc_recall, eval_acc_haming))
        # except:
        #     torch.save(model, configs.model_data_path + 'cpc_lstm.model')
        #     break

        last_epoch = i + 1
        torch.cuda.synchronize()
        times.append(time.time() - start)

    if last_epoch == epoch:
        torch.save(model, model_data_path + end)
    print("\ntimecost:", times, "\n")


if __name__ == '__main__':
    main()
