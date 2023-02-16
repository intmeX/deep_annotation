import requests
from bs4 import BeautifulSoup
import MySQLdb
import re
# from MySQLdb.cursors import Cursor

alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def initilize(dataBase):
    database = {
        "host": "127.0.0.1",
        "database": dataBase,
        "user": "root",
        "password": "1336296540.",
        "charset": "utf8mb4"
    }
    db = MySQLdb.connect(**database)
    return db


def insert_code(db, dicc):
    cc = "'{}', " * 5
    cc = cc.format(dicc["ID"], dicc["problem"], dicc["submit"], dicc["accept"], dicc["address"])
    cc = cc[:-2]
    buff = """
    insert into `poj`(`ID`, `problem`, `submit`, `accept`, `address`)
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
    url += str(page)
    responses = requests.get(url=url)
    soup = BeautifulSoup(responses.text, "lxml")
    tr = soup.find_all("tr", align="center")
    for i in tr:
        aa = {}
        aa["ID"] = i.contents[0].text.strip()
        aa["problem"] = i.contents[1].text.replace("\"", "_").replace("\\", "_").replace("'", "-").strip()
        aa["accept"] = i.contents[2].contents[1].text
        aa["submit"] = i.contents[2].contents[3].text
        aa["address"] = "http://poj.org/problem?id=" + aa["ID"]
        print(aa)
        insert_code(db, aa)
        print("Successfully insert one lines data into the table!")


url1 = "http://poj.org/problemlist?volume="

dataBase = "code"
db1 = initilize(dataBase)

for i in range(1, 32):
    get_im(url1, db1, page=i)
