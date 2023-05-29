import os
import requests
import bs4
import re


def get_soup(url):
    """
    获取
    """
    r = requests.get(url)
    r.encoding = 'utf-8'  # 用utf-8解码文档
    rt = r.text
    soup = bs4.BeautifulSoup(rt, 'lxml')
    return soup


if __name__ == '__main__':
    filepath = os.path.join(os.getcwd(), "url_text.txt")# 保存在当前路径
    with open(filepath, 'w', encoding='utf-8') as f2:
        f2.writelines("# url text\n")
        f2.close()
    url = "https://www.cib.com.cn/cn/index.html"
    soup = get_soup(url)
    soup = soup.find("div", {"id": "panel-person"})
    urllist = soup.find_all("a")

    for i in urllist:
        href = i['href']
        textname = i.text

        if textname == '':
            # dubug see i
            textname = i.contents[0]['alt']
        textname = re.sub('[\/:*?"<>|]\s', '-', textname)  # 使 文件命名 符合命名规定
        textname = re.su = re.sub(r'\s', "", textname)      # 删除文件名中的空白字符
        with open(filepath, 'a+', encoding='utf-8') as f:
            if "http" not in href:
                f.write(f'https://www.cib.com.cn{href}\t{textname} \n')
            else:
                f.write(f'{href}\t{textname} \n')
            f.close()
        print("https://www.cib.com.cn/cn"+href, textname)
    print("下载完成")