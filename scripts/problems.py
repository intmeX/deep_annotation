import os
import re
import copy
import html
import hashlib
import requests
from bs4 import BeautifulSoup
from scripts import XML
from scripts import config


def SavingText(set_name, ID, name, text_str):
    saving_dir = "./results/" + set_name + "/" + str(ID)
    fname = hashlib.md5(bytes(set_name + name.strip(), encoding="utf-8")).hexdigest()
    # text_str.replace("\n", "\r\n")
    with open(saving_dir + "/" + fname + ".txt", "w", encoding="gbk", errors="ignore") as f:
        f.write(text_str)


def Gather(p):
    set_name = p['oj']
    ID = p['ID']
    name = p['problem']
    url = p['address']
    header = {"User-Agent": config.get_ua()}
    try:
        response = requests.get(url, headers=header, timeout=10)
    except Exception:
        print("HTTP_ERROR in " + set_name + ' ' + str(ID))
        return None
    des_html = response.text
    soup = BeautifulSoup(des_html, "lxml")
    p['stmt'] = ''
    p['input_stmt'] = ''
    p['output_stmt'] = ''
    p['sample'] = ''
    p['note'] = ''
    if set_name == "hdu":
        tds = soup.find_all('td', align='center')
        soup = None
        for i in tds:
            if i.contents[0].name == 'h1':
                soup = i
                break
        if not soup:
            return None
        images = soup.find_all("img")
        if len(images) > 0:
            return None
        i = 0
        p['sample'] = '[Input]\n'
        while i < len(soup.contents) - 1:
            if soup.contents[i].text.strip() == 'Problem Description':
                i += 1
                while i < len(soup.contents) and \
                        (
                            'div' != soup.contents[i].name or
                            'class' not in soup.contents[i].attrs or
                            (
                                'panel_content' != soup.contents[i].attrs['class'] and
                                'panel_content' not in soup.contents[i].attrs['class']
                            )
                        ):
                    i += 1
                if i == len(soup.contents):
                    break
                p['stmt'] = soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Input':
                i += 1
                while i < len(soup.contents) and \
                        (
                                'div' != soup.contents[i].name or
                                'class' not in soup.contents[i].attrs or
                                (
                                        'panel_content' != soup.contents[i].attrs['class'] and
                                        'panel_content' not in soup.contents[i].attrs['class']
                                )
                        ):
                    i += 1
                if i == len(soup.contents):
                    break
                p['input_stmt'] = soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Output':
                i += 1
                while i < len(soup.contents) and \
                        (
                                'div' != soup.contents[i].name or
                                'class' not in soup.contents[i].attrs or
                                (
                                        'panel_content' != soup.contents[i].attrs['class'] and
                                        'panel_content' not in soup.contents[i].attrs['class']
                                )
                        ):
                    i += 1
                if i == len(soup.contents):
                    break
                p['output_stmt'] = soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Sample Input':
                i += 1
                while i < len(soup.contents) and \
                        (
                                'div' != soup.contents[i].name or
                                'class' not in soup.contents[i].attrs or
                                (
                                        'panel_content' != soup.contents[i].attrs['class'] and
                                        'panel_content' not in soup.contents[i].attrs['class']
                                )
                        ):
                    i += 1
                if i == len(soup.contents):
                    break
                p['sample'] += soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Sample Output':
                i += 1
                while i < len(soup.contents) and \
                        (
                                'div' != soup.contents[i].name or
                                'class' not in soup.contents[i].attrs or
                                (
                                        'panel_content' != soup.contents[i].attrs['class'] and
                                        'panel_content' not in soup.contents[i].attrs['class']
                                )
                        ):
                    i += 1
                if i == len(soup.contents):
                    break
                p['sample'] += '\n[Output]\n' + soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Hint':
                i += 1
                while i < len(soup.contents) and \
                        (
                                'div' != soup.contents[i].name or
                                'class' not in soup.contents[i].attrs or
                                (
                                        'panel_content' != soup.contents[i].attrs['class'] and
                                        'panel_content' not in soup.contents[i].attrs['class']
                                )
                        ):
                    i += 1
                if i == len(soup.contents):
                    break
                p['note'] = soup.contents[i].text.strip()
            i += 1
    elif set_name == "poj":
        soup = soup.find("table", background="images/table_back.jpg")
        if not soup:
            return None
        soup = soup.find('td')
        if not soup:
            return None
        images = soup.find_all("img")
        if len(images) > 0:
            return None
        i = 0
        p['sample'] = '[Input]\n'
        while i < len(soup) - 1:
            if soup.contents[i].text.strip() == 'Description':
                i += 1
                p['stmt'] = soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Input':
                i += 1
                p['input_stmt'] = soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Output':
                i += 1
                p['output_stmt'] = soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Sample Input':
                i += 1
                p['sample'] += soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Sample Output':
                i += 1
                p['sample'] += '\n[Output]\n' + soup.contents[i].text.strip()
            elif soup.contents[i].text.strip() == 'Hint':
                i += 1
                p['note'] = soup.contents[i].text.strip()
            i += 1
    else:
        soup = soup.find('div', class_='problem-statement')
        if not soup:
            return None
        if len(soup.contents) < 5:
            return None
        images = soup.find_all("img")
        if len(images) > 0:
            return None
        p['stmt'] = soup.contents[1].text.strip()
        p['input_stmt'] = soup.contents[2].text.strip()
        p['output_stmt'] = soup.contents[3].text.strip()
        p['sample'] = soup.contents[4].text.strip()
        if len(soup.contents) >= 6:
            p['note'] = soup.contents[5].text.strip()
    p['stmt'] = html.unescape(p['stmt'])
    p['input_stmt'] = html.unescape(p['input_stmt'])
    p['output_stmt'] = html.unescape(p['output_stmt'])
    p['sample'] = html.unescape(p['sample'].replace('\r\n', '\n'))
    p['note'] = html.unescape(p['note'])
    return p


def save(problems):
    if not os.path.exists(config.base_path):
        os.mkdir(config.base_path)
    XML.write_xml(config.base_path + 'problems.xml', problems, entity_name='problem')


if __name__ == "__main__":
    # Gather("hdu", 4756, "Install Air Conditioning", "http://acm.hdu.edu.cn/showproblem.php?pid=4756")
    problems = {'problem': []}
    pro_list = XML.read_xml(config.problem_list_path + r".*")
    # print(str(pro_list)[:2000])
    # continue
    # fl = True
    total = 0
    exception = None
    for i in pro_list['problem']:
        # if i["ID"] == "999E":
        #     break
        # if k == "codeforces" and fl:# or (i["ID"] >= 2783 and k == "poj") or (i["ID"] >= 1924 and k == "hdu")
        # Gather(k, i["ID"], i["problem"], i["address"].replace("ml", "com"))
        # if i['oj'] != 'poj':
        #     continue
        try:
            res = Gather(i)
        except Exception as e:
            save(problems)
            raise e
        if res:
            problems['problem'].append(res)
            total += 1
            print('Successfully get the problem content. ID:{} total:{}'.format(res['ID'], total))

    save(problems)
    # print("These are short errors list:")
    # for i in short_errors:
    #     print(i)
    # print("These are other errors list:")
    # for i in other_errors:
    #     print(i)

