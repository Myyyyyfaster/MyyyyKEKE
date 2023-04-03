import requests
from bs4 import BeautifulSoup
import numpy

# print('百年好合\n意思:')


def niuib():
    URL='https://dict.baidu.com/s?wd=灰头土面'
    header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69'}

    html=requests.get(URL,headers=header)
    soup=BeautifulSoup(html.text,'html.parser')
    word = soup.select('#jielong-wrapper > div.tab-content > a:nth-child(2)')[0]
    word=word.text
    print(word+'师范')

    print('-----------------------------')


def real():
    num2 = []
    num2.append('五光十色')
    print(num2[0])
    for i in range(20):
        URL = 'https://dict.baidu.com/s?wd='
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69'}

        html = requests.get(URL,params={'wd':num2[-1]}, headers=header)
        soup = BeautifulSoup(html.text, 'html.parser')
        word=soup.select('#jielong-wrapper > div.tab-content > a:nth-child(2)')[0]
        means=soup.select('#basicmean-wrapper > div:nth-child(2) > dl > dd > p')[0]

        print(means.text)
        print(word.text)

        num2.append(word.text)
        print(num2)


def real2():
    num2 = []
    num2.append('五光十色')
    result=['五光十色']
    for i in range(5):
        URL = 'https://dict.baidu.com/s?wd='
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69'}

        html = requests.get(URL,params={'wd':result[-1]}, headers=header)
        soup = BeautifulSoup(html.text, 'html.parser')
        word=soup.select('#jielong-wrapper > div.tab-content')[0:50]
        for word in word:
            print(word.get_text())

            num2.append(word.get_text())
        print(num2)
        result=[(s.strip()) for s in num2[1].split('\n')[1:-1]]
        print(result)
    print(len(result))





if __name__=='__main__':
    # niuib()
    real()
    # real2()