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

# 定义分词器
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
embedding = KeyedVectors.load_word2vec_format(model_data_path + 'GoogleNews-vectors-negative300.bin', binary=True)
# embedding = torch.load(config.model_data_path + 'GoogleNews-vectors-negative300.model').to(device)


# 自定义Dataset子类
class CpcText(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ID = self.dataset[idx]["ID"]
        desc = self.dataset[idx]["desc"]
        sample = {'ID': ID, "desc": desc}
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
    res['ID'] = data['ID']
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
    data_test = DataLoader(CpcText(data), batch_size=batch_size, num_workers=0)
    return data_test


def test(model, dataloader, thr=0.7):
    global num_classes
    global device
    global batch_size
    res = dict()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            desc = batch["desc"]
            # label = torch.tensor(batch["label"], dtype=torch.long).to(device)
            ID = batch["ID"]
            prob = model(desc)
            p = prob > thr
            for j in range(batch_size):
                idx = ID['ID'][j]
                res[idx] = ""
                for k in range(num_classes):
                    if p[j][k] == 1:
                        res[idx] += str(k) + ','
                if len(res[idx]) > 0:
                    res[idx] = res[idx][:-1]
    return res


def main():
    test_loader = data_prepare()
    print(test_loader.dataset[2])
    # model = lstm_ml.LSTM(None, vocab_len, vec_dim, hidden_dim, num_layers, num_classes=num_classes).to(device)
    model = torch.load(model_data_path + model_path)
    # optimizer = Adam(lr=1e-4, eps=1e-8, weight_decay=0.01)
    # 参考博客 一是去掉无用的设置 而是构造字典列表以使AdamW可以接受该参数

    last_epoch = 0

    torch.cuda.synchronize()
    start = time.time()
    res = test(model, test_loader)
    end = time.time()
    test_time = end - start
    torch.cuda.synchronize()
    with open(results_path + str(end) + '.txt', 'w') as f:
        fstr = ''
        for i in res:
            fstr += i + ' ' + res[i] + '\n'
        f.write(fstr)

    print("\ntimecost:", test_time, "\n")
    print("\nthe result has been saved in " + str(end) + '.txt\n')


if __name__ == '__main__':
    main()

