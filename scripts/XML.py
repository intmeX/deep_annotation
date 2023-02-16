from lxml import html
from xml.parsers.expat import ParserCreate

etree = html.etree


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
    c = etree.parse(path).getroot()
    return {'problems': etree2dict(c)}


def write_xml(path, c):
    root = etree.ElementTree(dict2etree(c, 'problems'))
    # dict2etree(c, 'problems').write(path, pretty_print=True)
    root.write(path, pretty_print=True)


if __name__ == '__main__':
    # write_xml("./1.xml", {'pro': [{'a': 1, 'b': 2}, {'a': 6, 'b': 2}]})
    print(read_xml('./1.xml'))


