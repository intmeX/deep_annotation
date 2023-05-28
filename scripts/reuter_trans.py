import pandas as pd
import random
from scripts import XML
from scripts import config
from transformers import BertTokenizerFast


def Reuters():
    train = pd.read_csv(config.reuters_path + 'reuters.csv', encoding='utf-8', usecols=['TOPICS', 'BODY'])
    train.rename(columns={'TOPICS': 'tag', 'BODY': 'text'}, inplace=True)
    c = None
    with open(config.reuters_path + 'all-topics-strings.lc.txt', 'r', encoding='utf-8') as f:
        c = f.readlines()
    random.seed(1)
    random.shuffle(c)
    tag2ind = dict()
    tag_info = []
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
    for i in range(len(c)):
        tag2ind[c[i].strip()] = i
        tag_info.append([i, c[i].strip(), 0])
    drop_list = []
    total_token = 0
    for index, i in train.iterrows():
        tags = str(i['tag']).split(',')
        tags_n = ''
        tokens = tokenizer.tokenize(str(i['text']))
        total_token += len(tokens)
        print("get doc {}".format(index))
        for j in tags:
            if j == '' or j not in tag2ind:
                continue
            tags_n += str(tag2ind[j]) + ','
            tag_info[tag2ind[j]][2] += 1
        if len(tags_n) > 0:
            tags_n = tags_n[:-1]
        else:
            drop_list.append(index)
        if len(str(i['text'])) < 5:
            drop_list.append(index)
            continue
        train.loc[index, 'tag'] = tags_n
    total = 0
    for i in sorted(tag_info, key=lambda x: x[2]):
        total += i[2]
        print(i)
    print('tag mean: {}'.format(total / 21578))
    print('token mean: {}'.format(total_token / 21578))
    # with open(config.reuters_path + 'all-topics.csv', 'w', encoding='utf-8') as f:
    #     f.write('ID,tag,num\n')
    #     for i in tag_info:
    #         f.write(','.join([str(j) for j in i]) + '\n')
    # train = train.drop(index=drop_list)
    # train.to_csv(config.reuters_path + 'train_no_empty.csv', index=False)


def ReutersEmpty():
    train = pd.read_csv(config.reuters_path + 'reuters.csv', encoding='utf-8', usecols=['TOPICS', 'BODY'])
    train.rename(columns={'TOPICS': 'tag', 'BODY': 'text'}, inplace=True)
    c = None
    with open(config.reuters_path + 'all-topics-strings.lc.txt', 'r', encoding='utf-8') as f:
        c = f.readlines()
    random.seed(1)
    random.shuffle(c)
    tag2ind = dict()
    tag_info = []
    for i in range(len(c)):
        tag2ind[c[i].strip()] = i
        tag_info.append([i, c[i].strip(), 0])
    drop_list = []
    for index, i in train.iterrows():
        tags = str(i['tag']).split(',')
        tags_n = ''
        print("get doc {}".format(index))
        for j in tags:
            if j == '' or j not in tag2ind:
                continue
            tags_n += str(tag2ind[j]) + ','
            tag_info[tag2ind[j]][2] += 1
        if len(tags_n) > 0:
            drop_list.append(index)
        elif len(str(i['text'])) < 5:
            drop_list.append(index)
            continue
    # with open(config.reuters_path + 'all-topics.csv', 'w', encoding='utf-8') as f:
    #     f.write('ID,tag,num\n')
    #     for i in tag_info:
    #         f.write(','.join([str(j) for j in i]) + '\n')
    train = train.drop(index=drop_list)
    train.to_csv(config.reuters_path + 'reuters_empty.csv', index=False)


def sta():
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
    train = pd.read_csv(config.reuters_path + 'train_no_empty.csv', encoding='utf-8')
    total = 0
    total_tags = 0
    for index, i in train.iterrows():
        tokens = tokenizer.tokenize(i['text'])
        total += len(tokens)
        total_tags += len(str(i['tag'])) / 2 + 1
        print("get doc {}".format(index))
    print('mean: {}'.format(total / len(train)))
    print('mean_num_tag: {}'.format(total_tags / len(train)))


if __name__ == '__main__':
    # Reuters()
    # sta()
    ReutersEmpty()

