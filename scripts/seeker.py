import requests
from bs4 import BeautifulSoup
# from .. import MysqL
import pandas as pd
from urllib import parse
import os



Repeat = set()
Search_num = set()
Proset = ["hdu", "poj", "codeforces"]
#


def Csdn(key_len, url):     # 对csdn博客的搜索
    ans = []
    Header = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "max-age=0",
            "cookie": "TY_SESSION_ID=00d40866-98f2-4aac-b351-bfa0e5b64b78; JSESSIONID=5378E31EE457E7B8403DE852990750F2; uuid_tt_dd=10_18645308390-1584514152943-517738; dc_session_id=10_1584514152943.307843; Hm_ct_6bcd52f51e9b3dce32bec4a3997715ac=6525*1*10_18645308390-1584514152943-517738; __gads=ID=c6ea9048cba4f3b6:T=1584514154:S=ALNI_MZAfJMwTjMfNhS5CPEO30pL4BO6Fw; Hm_up_6bcd52f51e9b3dce32bec4a3997715ac=%7B%22islogin%22%3A%7B%22value%22%3A%220%22%2C%22scope%22%3A1%7D%2C%22isonline%22%3A%7B%22value%22%3A%220%22%2C%22scope%22%3A1%7D%2C%22isvip%22%3A%7B%22value%22%3A%220%22%2C%22scope%22%3A1%7D%7D; dc_sid=384bd74d519b2d6a40ec8559a9fefe20; c_ref=https%3A//www.baidu.com/link; c_first_ref=www.baidu.com; c_first_page=https%3A//www.csdn.net/; Hm_lvt_6bcd52f51e9b3dce32bec4a3997715ac=1588760018,1589098015,1590111666,1590281907; searchHistoryArray=%255B%2522hdu1156%2522%252C%2522c%252B%252B%2522%255D; c_utm_term=hdu1156; c_utm_medium=distribute.pc_search_result.none-task-blog-2%7Eall%7Esobaiduweb%7Edefault-2-83243158; SESSION=c16f96bd-8b6c-45ca-9411-c3ec97492f37; c-toolbar-writeguide=1; c-login-auto=1; dc_tos=qat9hz; Hm_lpvt_6bcd52f51e9b3dce32bec4a3997715ac=1590282072",
            "referer": "https://www.csdn.net/",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-site",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36"}
    responses = requests.get(url=url, headers=Header)
    soup = BeautifulSoup(responses.text, "lxml")
    dls = soup.find_all("dl", class_="search-list J_search")
    for i in dls:
        Title_add = i.find("a")
        get_url = Title_add.attrs["href"].split("?")[0]     # 问号有可能影响到正确的题目，但是一般情况下不会
        if get_url in Repeat:
            Search_num.add(get_url)
            continue
        Repeat.add(get_url)

        key_Length1 = 0
        ems1 = Title_add.find_all("em")
        for j in ems1:
            key_Length1 += len(j.text)

        if key_Length1 >= key_len:
            abstract = i.find("dd", class_="search-detail").text
            ans.append([get_url, Title_add.text, abstract])

    return ans


def Cnblogs(key_len, url):      # 对cnblogs博客的搜索
    ans = []
    Header = {"content-encoding": "gzip",
            "content-type": "text/html; charset=utf-8",
            "status": "200",
            "vary": "Accept-Encoding",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "max-age=0",
            "cookie": "_ga=GA1.2.1780921277.1584516779; __gads=ID=5a737caad2a25e0e:T=1584516779:S=ALNI_MbuCGItmS83nVgsUJIux-yvztZ41g; UM_distinctid=171e7b0dbf5280-016384ea5a94d3-6373664-144000-171e7b0dbf64b0; _gid=GA1.2.1284251240.1590459322; DetectCookieSupport=OK; __utma=59123430.1780921277.1584516779.1590459330.1590459330.1; __utmc=59123430; __utmz=59123430.1590459330.1.1.utmcsr=cnblogs.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; .AspNetCore.Session=CfDJ8B9DwO68dQFBg9xIizKsC6SYG8%2BxDbkUAN6u9Tl9r7mDRbO9aSWLlKOXMENrk2wzS6Cl5UJw6HQkq%2FYUUKLAVhUF0zqUYwznQjzk8gWVzZseCZRFI%2BsGnqYYY%2F2GCeHQk9H3LaGDE6tAoaIxvD0XWkGH4wVdYltCQDdkSuiV6blo; SBNoRobotCookie=CfDJ8B9DwO68dQFBg9xIizKsC6SCgh3uwpW3BJqb9lyxAqvFxyNDjtf3d0TKus-vcPd8KxHmEDdgoAK3RQcYehoJKq1IBsw8TUGAEerjR6df3p9oOEprMKn9SBryPRAXqqAbwg; __utmb=59123430.5.10.1590459330",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36"}
    responses = requests.get(url=url, headers=Header)
    soup = BeautifulSoup(responses.text, "lxml")
    divs = soup.find_all("div", class_="searchItem")
    for i in divs:
        Title_add = i.find("a", target="_blank")
        get_url = Title_add.attrs["href"]
        if get_url in Repeat:
            Search_num.add(get_url)
            continue
        Repeat.add(get_url)

        key_Length1 = 0
        ems1 = Title_add.find_all("strong")
        for j in ems1:
            key_Length1 += len(j.text)

        if key_Length1 >= key_len:
            abstract = i.find("span", class_="searchCon").text
            ans.append([get_url, Title_add.text, abstract])

    return ans


if __name__ == "__main__":
    Info = ""
    Searcher = {}
    results = {}

    with open("../files/search_urls", "r") as f:
        for i in f.readlines():
            Searcher[i.split()[0]] = [i.split()[1], i.split()[2]]

    for k in Proset:
        csv_path = "../" + k + ".csv"
        df = pd.read_csv(csv_path, encoding="gbk")
        results_df = pd.DataFrame(columns=["ID", "problem", "name", "address", "search", "visit", "recommend", "feedback"])
        for index, o in df.iterrows():
            Repeat = set()
            ID = str(o["ID"]).strip()
            problem = o["problem"].strip()
            Info = k + ' ' + ID
            k_len = len(Info.replace(" ", ""))
            for i in Searcher.keys():
                results[i] = []
                new_url = Searcher[i][0] + parse.quote(Info) + Searcher[i][1]
                if i == "CSDN":
                    Gain = Csdn(k_len, new_url)
                    if Gain:
                        results[i] += Gain
                if i == "cnblogs":
                    Gain = Cnblogs(k_len, new_url)
                    if Gain:
                        results[i] += Gain
            Info += ' ' + problem
            k_len = len(Info.replace("+", "").replace("-", "").replace(" ", ""))
            for i in Searcher.keys():
                new_url = Searcher[i][0] + parse.quote(Info) + Searcher[i][1]
                if i == "CSDN":
                    Gain = Csdn(k_len, new_url)
                    if Gain:
                        results[i] += Gain
                if i == "cnblogs":
                    Gain = Cnblogs(k_len, new_url)
                    if Gain:
                        results[i] += Gain
            for i in Searcher.keys():
                for j in results[i]:
                    s_n = 0
                    if j[0] in Search_num:
                        s_n = 1
                    newdf = pd.DataFrame({"ID": ID,
                                          "problem": problem,
                                          "name": j[1],
                                          "address": j[0],
                                          "search": s_n,
                                          "visit": 0,
                                          "recommend": 0,
                                          "feedback": 0}, index=[1])
                    results_df = results_df.append(newdf, ignore_index=True)
            print("Successfully get the solution's information of " + k + " " + ID)
        results_df.to_csv("../files/csv/" + k + "_article.csv", index=False, sep=',', encoding="utf-8")
        # results_df.to_csv("../files/csv/" + k + "_article2.csv", index=False, sep=',', encoding="gbk")


