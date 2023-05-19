import pandas as pd
import random
from scripts import XML
from scripts import config
from transformers import BertTokenizerFast


def AGNews():
    train = pd.read_csv(config.AGNews_path + 'train.csv', encoding='utf-8')
    test = pd.read_csv(config.AGNews_path + 'test.csv', encoding='utf-8')
    for index, i in train.iterrows():
        tags = int(i['tag'])
        print("get doc {}".format(index))
        train.loc[index, 'tag'] = str(tags - 1)
        train.loc[index, 'text'] = i['title'] + ' ' + i['text']
    for index, i in test.iterrows():
        tags = int(i['tag'])
        test.loc[index, 'tag'] = str(tags - 1)
        test.loc[index, 'text'] = i['title'] + ' ' + i['text']
    train.to_csv(config.AGNews_path + 'train_1.csv', index=False)
    test.to_csv(config.AGNews_path + 'test_1.csv', index=False)


def sta():
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
    train = pd.read_csv(config.AGNews_path + 'train_1.csv', encoding='utf-8')
    total = 0
    for index, i in train.iterrows():
        tokens = tokenizer.tokenize(i['text'])
        total += len(tokens)
        print("get doc {}".format(index))
    print('mean: {}'.format(total / 120000))


if __name__ == '__main__':
    # AGNews()
    sta()
