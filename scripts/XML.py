import os
import re
# import copy
from lxml import html
etree = html.etree


def path_list(path):
    l = len(path)
    while path[l - 1] != '/':
        l -= 1
    p = re.compile(path[l:])
    res = []
    for f in os.listdir(path[:l]):
        if p.match(f):
            res.append(path[:l] + f)
    return res


def etree2dict(c):
    if len(c) == 0:
        return c.text
    else:
        res = {}
        for i in c:
            item = etree2dict(i)
            if i.tag in res:
                if not isinstance(res[i.tag], list):
                    res[i.tag] = [res[i.tag]]
                res[i.tag].append(item)
            else:
                res[i.tag] = item
        return res


def dict2etree(c, k):
    node = etree.Element(str(k))
    if isinstance(c, list):
        raise Exception("cannot recognize the dict at \'" + str(c)[: 10] + "\'")
    elif isinstance(c, dict):
        for key, val in c.items():
            if isinstance(val, list):
                for i in val:
                    node.append(dict2etree(i, key))
            else:
                node.append(dict2etree(val, key))
    else:
        node.text = str(c)
    return node


def read_xml(path):
    res = {}
    for i in path_list(path):
        tree = etree.parse(i).getroot()
        dic = etree2dict(tree)
        if len(res) == 0:
            res = dic
        else:
            for j in res:
                res[j].extend(dic[j])
    return res


def write_xml(path, c):
    root = etree.ElementTree(dict2etree(c, 'problems'))
    # dict2etree(c, 'problems').write(path, pretty_print=True)
    root.write(path, pretty_print=True)


if __name__ == '__main__':
    write_xml("./1.xml", {'pro': [{'a': 1, 'b': 2}, {'a': 6, 'b': 2}]})
    write_xml("./2.xml", {'pro': [{'a': 2, 'b': 3}, {'a': 7, 'b': 3}]})
    print(read_xml(r'./\d+\.xml'))


