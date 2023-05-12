import pandas as pd
from scripts import XML
from scripts import config


def AAPD():
    train = pd.read_csv(config.AAPD_path + 'aapd_train.tsv', encoding='utf-8', sep='\t', names=['tag', 'text'])
    val = pd.read_csv(config.AAPD_path + 'aapd_validation.tsv', encoding='utf-8', sep='\t', names=['tag', 'text'])
    test = pd.read_csv(config.AAPD_path + 'aapd_test.tsv', encoding='utf-8', sep='\t', names=['tag', 'text'])
    for index, i in train.iterrows():
        tag = str(i['tag'])
        tag_n = ''
        j = 0
        while j < len(tag):
            if tag[j] == '1':
                tag_n += str(j) + ','
            j += 1
        if len(tag_n) > 0:
            tag_n = tag_n[:-1]
        train.loc[index, 'tag'] = tag_n
    for index, i in val.iterrows():
        tag = str(i['tag'])
        tag_n = ''
        j = 0
        while j < len(tag):
            if tag[j] == '1':
                tag_n += str(j) + ','
            j += 1
        if len(tag_n) > 0:
            tag_n = tag_n[:-1]
        val.loc[index, 'tag'] = tag_n
    for index, i in test.iterrows():
        tag = str(i['tag'])
        tag_n = ''
        j = 0
        while j < len(tag):
            if tag[j] == '1':
                tag_n += str(j) + ','
            j += 1
        if len(tag_n) > 0:
            tag_n = tag_n[:-1]
        test.loc[index, 'tag'] = tag_n
    aapd_train = pd.concat([train, val], axis=0, ignore_index=True)
    aapd_train.to_csv(config.AAPD_path + 'aapd_train.csv', index=False)
    test.to_csv(config.AAPD_path + 'aapd_test.csv', index=False)


if __name__ == '__main__':
    AAPD()
