import pandas as pd
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


if __name__ == '__main__':
    pack_tag()
