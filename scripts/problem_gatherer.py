import os
import re
import random
import urllib
import urllib.request
import hashlib
import requests
import pandas as pd
from bs4 import BeautifulSoup


Proset = ["codeforces"]#"hdu", "poj",
UA = ["Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
      "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
      "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
      "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
      "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11"]

short_errors = []
other_errors = []


def SavingText(set_name, ID, name, text_str):
    saving_dir = "./results/" + set_name + "/" + str(ID)
    fname = hashlib.md5(bytes(set_name + name.strip(), encoding="utf-8")).hexdigest()
    # text_str.replace("\n", "\r\n")
    with open(saving_dir + "/" + fname + ".txt", "w", encoding="gbk", errors="ignore") as f:
        f.write(text_str)


def SavingImage(set_name, ID, urls):
    saving_dir = "./results/" + set_name + "/" + str(ID) + "/images"
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    cnt = 0
    for i in urls:
        cnt += 1
        fname = set_name + str(ID) + "_" + str(cnt) + "." + i.split(".")[-1]
        try:
            urllib.request.urlretrieve(i, saving_dir + "/" + fname)
        except urllib.error.ContentTooShortError:
            print("ContentTooShortError in the " + set_name + str(ID))
            short_errors.append(set_name + str(ID))
        except:
            print("OtherError in the " + set_name + str(ID))
            other_errors.append(set_name + str(ID))



def SavingFormula(set_name, ID, numm, ML_str):
    saving_dir = "./results/" + set_name + "/" + str(ID) + "/mathMLs"
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    fname = set_name + str(ID) + "_" + str(numm)
    with open(saving_dir + "/" + fname + ".xml", "w", encoding="utf-8") as f:
        f.write(ML_str)


def Gather(set_name, ID, name, url):
    saving_dir = "./results/" + set_name + "/" + str(ID)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    description = BeautifulSoup()
    header = {"User-Agent": UA[random.randint(0, 5)]}
    try:
        response = requests.get(url, headers=header)
    except:
        print("HTTP_ERROR in " + set_name + str(ID))
        return
    des_html = response.text
    soup = BeautifulSoup(des_html, "lxml")
    sp = 0
    if set_name == "hdu":
        table = soup.find("table")
        description = table.contents[-3]
        sp = response.text.find("<tr><td align=center><h1")

    elif set_name == "poj":
        description = soup.find("table", background="images/table_back.jpg")
        sp = response.text.find("<table border=0 width=100% background=images/table_back.jpg>")

    else:
        description = soup.find("div", class_="problem-statement")
        sp = response.text.find("<div class=\"problem-statement\">")

    if description is None:
        return
    else:
        problem_text = description.text
    images = description.find_all("img")
    image_urls = []
    ll = 0
    for i in images:
        image_url = i.get("src")
        if image_url[0] == '/':
            image_url = image_url[1:]
        if set_name == "hdu":
            im_ind = image_url.find("data")
            image_urls.append("http://acm.hdu.edu.cn/" + image_url[im_ind:])
        elif set_name == "poj" and image_url[0] != 'f':
            image_urls.append("http://poj.org/" + image_url)
        elif set_name == "codeforces":
            image_urls.append(image_url)
        sp = des_html.find("<img", sp)
        sp2 = des_html.find(">", sp) + 1
        img_tag = ""
        if ll < len(image_urls):
            ll += 1
            img_tag = "\nimg{" + str(len(image_urls)) + "}\n"
        des_html = des_html[:sp] + img_tag + des_html[sp2:]
    if image_urls:
        SavingImage(set_name, ID, image_urls)
    # formulas = description.find_all("span", class_="MathJax")  # 如果是Latex源码 则不需要单独存储
    # cnt = 0
    # for i in formulas:
    #     cnt += 1
    #     SavingFormula(set_name, ID, cnt, i.get("data-mathml"))
    if set_name == "hdu":
        table = soup.find("table")
        description = table.contents[-3]

    elif set_name == "poj":
        des_html = des_html.replace("<p class=\"pst\">Input", "<p class=\"pst\">${I_d}Input")
        des_html = des_html.replace("<p class=\"pst\">Output", "<p class=\"pst\">${O_d}Output")
        des_html = des_html.replace("Sample Input</p>", "${I}Sample Input</p>")
        des_html = des_html.replace("Sample Output</p>", "${O}Sample Output</p>")

    else:
        des_html = des_html.replace("<div class=\"input-specification\">", "<div class=\"input-specification\">${I_d}")
        des_html = des_html.replace("<div class=\"output-specification\">", "<div class=\"input-specification\">${O_d}")
        des_html = des_html.replace("<div class=\"input\">", "<div class=\"input\">${I}")
        des_html = des_html.replace("<div class=\"output\">", "<div class=\"output\">${O}")

    des_html = des_html.replace("<br>", "\n")
    des_html = des_html.replace("</p>", "</p>\n")
    des_html = des_html.replace("</pre>", "</pre>\n")
    des_html = des_html.replace("<br />", "\n")
    des_html = des_html.replace("</div>", "</div>\n")
    des_html = des_html.replace("</span>", "</span>\n")
    soup = BeautifulSoup(des_html, "lxml")
    if set_name == "hdu":
        table = soup.find("table")
        description = table.contents[-3]

    elif set_name == "poj":
        description = soup.find("table", background="images/table_back.jpg")

    else:
        description = soup.find("div", class_="problem-statement")

    result_text = description.text
    result_text = re.sub("\n+", '\n', result_text)
    SavingText(set_name, ID, name, result_text)
    print("Successfully get the " + set_name, str(ID))


if __name__ == "__main__":
    # Gather("hdu", 4756, "Install Air Conditioning", "http://acm.hdu.edu.cn/showproblem.php?pid=4756")
    for k in Proset:
        df = pd.read_csv(k + ".csv", encoding="gbk")
        fl = True
        for index, i in df.iterrows():
            if i["ID"] == "999E":
                break
            if k == "codeforces" and fl:# or (i["ID"] >= 2783 and k == "poj") or (i["ID"] >= 1924 and k == "hdu")
                Gather(k, i["ID"], i["problem"], i["address"].replace("ml", "com"))

    print("These are short errors list:")
    for i in short_errors:
        print(i)
    print("These are other errors list:")
    for i in other_errors:
        print(i)
