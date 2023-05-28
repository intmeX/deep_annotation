import os
import numpy as np
import pandas as pd
import time
import torch
import random
from transformers import BertForSequenceClassification, \
    BertTokenizerFast, BertConfig
from torch.utils.data import Dataset, DataLoader
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
    data = get_data(data_path)
    # random.seed(1)
    # random.shuffle(data)
    validate_loader = DataLoader(CpcText(data), batch_size=1, num_workers=0)
    return validate_loader, data


def run_predict(model, dataloader, tokenizer, data, thr=threshold):
    global num_classes
    global device
    global single_label
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            desc = batch[feature]
            # label = torch.tensor(batch[label], dtype=torch.long).to(device)

            tokenizer_res = tokenizer(
                desc,
                max_length=max_length,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                return_token_type_ids=True,
                return_attention_mask=True,
                verbose=True,
                return_tensors="pt")
            tokenizer_res = tokenizer_res.to(device)

            output = model(**tokenizer_res)
            prob = output[0]

            pred = prob > thr
            res_str = ''
            for j in range(num_classes):
                if pred[0][j] > 0:
                    res_str += str(j) + ','
            if len(res_str) > 0:
                res_str = res_str[:-1]
            if single_label:
                pred = prob.argmax(dim=1)[0]
                res_str = str(pred.item())
            data[i]['pre_label'] = res_str
            print("text {:>5} : {}".format(i, res_str))


def main():
    global last_epoch
    global device
    global end
    if model_path == '_':
        raise Exception('No model path in this task')

    # 定义分词器
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')

    validate_loader, pre_dict = data_prepare()
    print(validate_loader.dataset[2])
    # BERT参数设置
    bert_config = BertConfig.from_pretrained(bert_name)
    bert_config.output_hidden_states = False
    bert_config.num_labels = num_classes
    bert_config.hidden_dropout_prob = hidden_dropout_prob
    # 模型加载
    # model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_data_path + model_path, config=bert_config)
    model = torch.load(model_data_path + model_path)
    model.to(device)

    times = []

    for i in range(epoch):
        torch.cuda.synchronize()
        start = time.time()
        # try:
        run_predict(model, validate_loader, tokenizer, pre_dict)
        # except:
        #     torch.save(model, configs.model_data_path + 'cpc_lstm.model')
        #     break

        torch.cuda.synchronize()
        times.append(time.time() - start)

    print("\ntimecost:", times, "\n")

    end = time.strftime("%Y%m%d%H%M%S", time.localtime())
    keys = pre_dict[0].keys()
    res = {i: [] for i in keys}
    for i in pre_dict:
        for j in keys:
            res[j].append(i[j])
    df = pd.DataFrame.from_dict(res)
    df.to_csv(results_path + end + '.csv', encoding='utf-8', index=False)
    print("\nthe result has been saved in " + end + '.csv\n')


if __name__ == '__main__':
    main()
