import pandas as pd
import random
from scripts import XML
from scripts import config


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
    for i in range(len(c)):
        tag2ind[c[i].strip()] = i
        tag_info.append([i, c[i].strip(), 0])
    for index, i in train.iterrows():
        tags = str(i['tag']).split(',')
        tags_n = ''
        for j in tags:
            if j == '' or j not in tag2ind:
                continue
            tags_n += str(tag2ind[j]) + ','
            tag_info[tag2ind[j]][2] += 1
        if len(tags_n) > 0:
            tags_n = tags_n[:-1]
        train.loc[index, 'tag'] = tags_n
    print(sorted(tag_info, key=lambda x: x[2]))
    with open(config.reuters_path + 'all-topics.csv', 'w', encoding='utf-8') as f:
        f.write('ID,tag,num\n')
        for i in tag_info:
            f.write(','.join([str(j) for j in i]) + '\n')
    train.to_csv(config.reuters_path + 'train.csv', index=False)


if __name__ == '__main__':
    Reuters()

