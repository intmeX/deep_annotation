import pandas as pd
import numpy as np
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, random_split, DataLoader
from gensim.models import word2vec
from gensim.models import KeyedVectors
from scripts import XML
from scripts import config


model_path = '../data/models/GoogleNews-vectors-negative300.bin'


if __name__ == '__main__':
    embedding = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print(embedding['<PAD>'])
    print(embedding['<UNK>'])
    # embedding1 = torch.nn.Embedding(3000000, 300)
    # 可训练参数转化
    # embedding1.weight = torch.nn.Parameter(torch.tensor(embedding.vectors), requires_grad=True)
    # print(embedding1('there'))
    # torch.save(embedding1, '../data/models/GoogleNews-vectors-negative300.model')
