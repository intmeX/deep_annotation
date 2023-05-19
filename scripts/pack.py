import pandas as pd
from scripts import XML
from scripts import config
from transformers import BertTokenizerFast


def pack_tag():
    tags = pd.read_csv(config.base_path + 'tags.csv', encoding='utf-8')
    tags_repr = pd.read_csv(config.base_path + 'representative_tags.csv', encoding='utf-8')
    problem_tag = pd.read_csv(config.base_path + 'problems_representatives.csv', encoding='utf-8')
    tag_p = dict()
    tag_new = dict()
    tags_dict = {'tag': []}
    problem_dict = dict()
    for index, i in tags.iterrows():
        tag_p[i['id']] = i['same']
    for index, i in tags_repr.iterrows():
        tag_new[i['id']] = index
        tags_dict['tag'].append({'id': index, 'tag': i['tag'], 'p': ''})
    for index, i in tags_repr.iterrows():
        tags_dict['tag'][index]['p'] = tag_new[i['p']]
    for index, i in problem_tag.iterrows():
        tag = str(tag_new[tag_p[i['tag']]])
        if i['problem'] in problem_dict:
            problem_dict[i['problem']] += ',' + tag
        else:
            problem_dict[i['problem']] = tag
    XML.write_xml(config.base_path + 'tags.xml', tags_dict, 'tag')
    problems = XML.read_xml(config.base_path + 'problems.xml')
    problems_with_tag = {'problem': []}
    problems_no_tag = {'problem': []}
    for i in problems['problem']:
        if i['ID'] in problem_dict:
            i['tag'] = problem_dict[i['ID']]
            problems_with_tag['problem'].append(i)
        else:
            problems_no_tag['problem'].append(i)
    XML.write_xml(config.base_path + 'problems_with_tag.xml', problems_with_tag, 'problem')
    XML.write_xml(config.base_path + 'problems_no_tag.xml', problems_no_tag, 'problem')
    pass


def tag_sta():
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
    problem = XML.read_xml(config.base_path + 'problems_with_tag50.xml')['problem']
    total = 0
    total_token = 0
    for i in problem:
        tokens = tokenizer.tokenize(str(i['stmt']))
        total_token += len(tokens)
        total += len(str(i['tag'])) // 2 + 1
        print("get doc {}".format(i['ID']))
    print('tag mean: {}'.format(total / 6841))
    print('token mean: {}'.format(total_token / 6841))


if __name__ == '__main__':
    # pack_tag()
    tag_sta()
