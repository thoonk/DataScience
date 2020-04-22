# csv.DicReader를 이용해 헤더를 key로 사용하는 딕셔너리로 읽어오기
# import csv
# with open('tab_delimited_stock_prices.txt')as f:
#     tab_reader = csv.reader(f, delimiter='\t')
#     for row in tab_reader:
#         data = row[0]
#         symbol = row[1]
#         closing_price = float(row[2])
#         print(data, symbol, closing_price)

# # 콜론 구분자 파일 출력
# with open ('colon_delimited_stock_prices.txt', 'w')as f:
#     f.write("""date:symbol:closing_price
#     6/20/2014:AAPL:90.91
#     6/20/2014:MSFT:41.68
#     6/20/2014:FB:64.5""")

# 웹 스크래핑
# from bs4 import BeautifulSoup
# import requests
# url = ("https://raw.githubusercontent.com/"
#        "joelgrus/data/master/getting-data.html")
# html = requests.get(url).text
# # (1) ULR의 Html 파일을 파싱 및 트리 생성
# soup = BeautifulSoup(html, 'html5lib')
# # (2) 첫 <p> 태그의 값 얻기
# first_paragraph = soup.find('p')        # or just soup.p
#
# # 첫번째 p를 찾아 라인을 리턴
# assert str(soup.find('p')) == '<p id="p1">This is the first paragraph.</p>'
# print(soup.find('p'))
# # p의 text 속성
# first_paragraph_text = soup.p.text
# first_paragraph_words = soup.p.text.split()
# assert first_paragraph_words == ['This', 'is', 'the', 'first', 'paragraph.']
# print(first_paragraph_words)
#
# first_paragraph_id = soup.p['id']
# first_paragraph_id2 = soup.p.get('id')
# assert first_paragraph_id == first_paragraph_id2 == 'p1'
#
# all_paragraphs = soup.find_all('p')
# paragraph_with_ids = [p for p in soup('p') if p.get('id')]
# assert len(all_paragraphs) == 2
# print(all_paragraphs)
# assert len(paragraph_with_ids) == 1
# print(paragraph_with_ids)

# 모듈 json을 통해 파싱 가능
# import json
# serialized = """{ "title" : "Data Science Book",
#                   "author" : "Joel Grus",
#                   "publicationYear" : 2019,
#                   "topics" : [ "data", "science", "data science"] }"""
#
# # parse the JSON to create a Python dict
# deserialized = json.loads(serialized)
# assert deserialized["publicationYear"] == 2019
# print(deserialized["publicationYear"])
# assert "data science" in deserialized["topics"]
# print(deserialized["topics"])

# 인증이 필요하지 않은 API 사용하기
# import requests, json
#
# github_user = "thoonk"
# endpoint = f"https://api.github.com/users/{github_user}/repos"
#
# repos = json.loads(requests.get(endpoint).text) # 
# print(requests.get(endpoint).text)
# from collections import Counter
# from dateutil.parser import parse
#
# dates = [parse(repo["created_at"]) for repo in repos]
# month_counts = Counter(date.month for date in dates)
# weekday_counts = Counter(date.weekday() for date in dates)
# print(month_counts)
# print(weekday_counts)

# 트위터 API 사용하기
# import os
#
# # Feel free to plug your key and secret in directly
# CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
# CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")
#
# import webbrowser
# from twython import Twython
#
# # Get a temporary client to retrieve an authentication url
# temp_client = Twython(CONSUMER_KEY, CONSUMER_SECRET)
# temp_creds = temp_client.get_authentication_tokens()
# url = temp_creds['auth_url']
#
# # Now visit that URL to authorize the application and get a PIN
# print(f"go visit {url} and get the PIN code and paste it below")
# webbrowser.open(url)
# PIN_CODE = input("please enter the PIN code: ")
#
# # Now we use that PIN_CODE to get the actual tokens
# auth_client = Twython(CONSUMER_KEY,
#                       CONSUMER_SECRET,
#                       temp_creds['oauth_token'],
#                       temp_creds['oauth_token_secret'])
# final_step = auth_client.get_authorized_tokens(PIN_CODE)
# ACCESS_TOKEN = final_step['oauth_token']
# ACCESS_TOKEN_SECRET = final_step['oauth_token_secret']
#
# # And get a new Twython instance using them.
# twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

