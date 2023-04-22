import pandas as pd
import random
from scripts import XML
from scripts import config


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


def show_tag():
    problem = XML.read_xml(config.base_path + 'problems_with_tag.xml')['problem']
    tag = XML.read_xml(config.base_path + 'tags.xml')['tag']
    for i in tag:
        i['num'] = 0
    for i in problem:
        tags = [int(j) for j in i['tag'].split(',')]
        for t in tags:
            tag[t]['num'] += 1
    return tag, problem


if __name__ == '__main__':
    tag, problem = show_tag()
    tag = sorted(tag, key=lambda x: x['num'])
    tag_50 = tag[-50:]
    random.shuffle(tag_50)
    tag_new = {}
    for i in range(50):
        print(tag_50[i])
        tag_new[int(tag_50[i]['id'])] = i
        tag_50[i]['id_50'] = i
    XML.write_xml(config.base_path + 'tags_50.xml', {'tag': tag_50}, 'tag')
    problem_new = []
    for i in problem:
        tags = [int(j) for j in i['tag'].split(',')]
        tags_r = ""
        for t in tags:
            if t in tag_new:
                if len(tags_r) == 0:
                    tags_r = str(tag_new[t])
                else:
                    tags_r += ',' + str(tag_new[t])
        if len(tags_r) > 0:
            i['tag'] = tags_r
            problem_new.append(i)
    XML.write_xml(config.base_path + 'problems_with_tag50.xml', {'problem': problem_new}, 'problem')
