import requests
from bs4 import BeautifulSoup
import MySQLdb
import re
# from MySQLdb.cursors import Cursor




def initilize(dataBase):
    database = {
        "host": "127.0.0.1",
        "database": dataBase,
        "user": "root",
        "password": "1336296540.",
        "charset": "gbk"
    }
    db = MySQLdb.connect(**database)
    return db


def insert_code(db, dicc):
    cc = "'{}', " * 5
    cc = cc.format(dicc["ID"], dicc["problem"], dicc["submit"], dicc["accept"], dicc["address"])
    cc = cc[:-2]
    buff = """
    insert into `hdu`(`ID`, `problem`, `submit`, `accept`, `address`)
    values({})
    """.format(cc)
    cursor = db.cursor()
    cursor.execute(buff)
    db.commit()


# def get_links(url, page=0):
#     if page != 1:
#         url += str(page)
#     responses = requests.get(url=url)
#     soup = BeautifulSoup(responses.text, "lxml")
#     links_tr = soup.find_all("tr")
#     aa = ["https://bj.lianjia.com" + div.a.get("href") for div in links_div]
#     return aa


def get_im(url, db, page=0):
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36",
        "cookies": "PHPSESSID=2l8tl612iopg1jqug4bfaeqvn6"}
    url += str(page)
    responses = requests.get(url=url, headers=header)
    soup = BeautifulSoup(responses.text, "lxml")
    tr = soup.find_all("table", class_="table_text")
    for i in tr[0].text.split(";"):
        if i == "" or i == " ":
            break
        cc = []
        aa = {}
        if i[0] == "P":
            cc = i.split(")p(")[1][: -1].split(",")
        else:
            cc = i[2: -1].split(",")
        aa["ID"] = cc[1]
        if aa["ID"] == "2171":
            a = 1
        if len(cc) >= 7:
            aa["problem"] = ("".join(cc[3: len(cc) - 2])).strip().replace("'", "-")
            aa["accept"] = cc[len(cc) - 2]
            aa["submit"] = cc[len(cc) - 1]
        else:
            aa["problem"] = cc[3].strip().replace("'", "-")
            aa["accept"] = cc[4]
            aa["submit"] = cc[5]
        aa["problem"] = aa["problem"][1: -1].replace("\"", "_").replace("\\", "_").strip()
        aa["address"] = "http://acm.hdu.edu.cn/showproblem.php?pid=" + aa["ID"]
        print(aa)
        insert_code(db, aa)
        print("Successfully insert one lines data into the table!")


url1 = "http://acm.hdu.edu.cn/listproblem.php?vol="

dataBase = "code"
db1 = initilize(dataBase)

for i in range(1, 59):
    get_im(url1, db1, page=i)
