import os
import re
import requests
from bs4 import BeautifulSoup
from scripts import XML
from scripts import config


def get_im(oj, page=0):
    header = {"User-Agent": config.get_ua()}
    url = config.prefix[oj] + str(page)
    try:
        responses = requests.get(url=url, timeout=10, headers=header)
    except requests.RequestException:
        return False
    soup = BeautifulSoup(responses.text, "lxml")
    BeautifulSoup.contains_replacement_characters = True
    problems = {'problem': []}
    if oj == 'codeforces':
        tr = soup.find_all("tr")
        co = 0
        for i in reversed(tr):
            co += 1
            if co == 1:
                continue
            if co == len(tr):
                break
            aa = dict()
            num = i.contents[1].text.strip()
            aa["ID"] = 'CF' + num
            info = re.sub("\n+", "\n", i.contents[3].text.strip()).split("\n")
            aa["problem"] = info[0].replace("'", "-").strip()
            aa["label"] = "".join(info[1:])
            aa["label"] = re.sub("[\r\t]", "", aa["label"])
            aa["label"] = re.sub(" +", " ", aa["label"])
            aa["difficulty"] = i.contents[7].text.strip()
            aa["accept"] = i.contents[9].text.strip().strip()[1:]
            cut = 0
            for j in range(len(num)):
                if num[j].isalpha():
                    cut = j
            aa["address"] = "https://codeforces.ml/problemset/problem/" + num[:cut] + "/" + num[cut:]
            aa['oj'] = oj
            print(aa)
            print("Successfully insert one lines data into the table! ID: {}, UAPOS:{}".format(aa['ID'], config.get_ua_pos()))
            problems['problem'].append(aa)
    elif oj == 'poj':
        tr = soup.find_all("tr", align="center")
        for i in tr:
            aa = dict()
            aa["ID"] = 'POJ' + i.contents[0].text.strip()
            aa["problem"] = i.contents[1].text.replace("\"", "_").replace("\\", "_").replace("'", "-").strip()
            aa["accept"] = i.contents[2].contents[1].text
            aa["submit"] = i.contents[2].contents[3].text
            aa["address"] = "http://poj.org/problem?id=" + i.contents[0].text.strip()
            aa['oj'] = oj
            print(aa)
            print("Successfully insert one lines data into the table! ID: {}, UAPOS:{}".format(aa['ID'], config.get_ua_pos()))
            problems['problem'].append(aa)
    elif oj == 'hdu':
        tr = soup.find("table", class_="table_text")
        local_text = tr.contents[1].text
        for i in local_text.split(");"):
            if i == "" or i == " ":
                break
            cc = []
            aa = dict()
            if i[0] == "P":
                continue
                # cc = i.split(")p(")[1][: -1].split(",")
            else:
                cc = i[2:].split(",")
            aa["ID"] = 'HDU' + cc[1]
            if len(cc) >= 7:
                aa["problem"] = ("".join(cc[3: len(cc) - 2])).strip().replace("'", "-")
                aa["accept"] = cc[len(cc) - 2]
                aa["submit"] = cc[len(cc) - 1]
            else:
                aa["problem"] = cc[3].strip().replace("'", "-")
                aa["accept"] = cc[4]
                aa["submit"] = cc[5]
            aa["problem"] = aa["problem"][1: -1].replace("\"", "_").replace("\\", "_").strip()
            aa["address"] = "http://acm.hdu.edu.cn/showproblem.php?pid=" + cc[1]
            aa['oj'] = oj
            print(aa)
            print("Successfully insert one lines data into the table! ID: {}, UAPOS:{}".format(aa['ID'], config.get_ua_pos()))
            problems['problem'].append(aa)
    else:
        raise Exception("No such OJ \'" + oj + "\'")
    if len(problems['problem']) == 0:
        return False
    if not os.path.exists(config.problem_list_path):
        os.makedirs(config.problem_list_path)
    XML.write_xml(config.problem_list_path + oj + '_' + str(page) + '.xml', problems)
    print('\nsave the file ' + config.problem_list_path + oj + '_' + str(page) + '.xml\n')
    return True


if __name__ == '__main__':
    for k in range(0, 3):
        i = 1
        while i <= config.page[k]:
            if not get_im(config.oj_name[k], i):
                i -= 1
            i += 1
