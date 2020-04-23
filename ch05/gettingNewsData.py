import requests
#과제
url = ('http://newsapi.org/v2/top-headlines?'
       'country=us&'
       'apiKey=19d99fa6c33f4138b6900efc72aa225c')
response = requests.get(url)
response_json = response.json()
#print(response.json())
print("totalResults = " + str(response_json["totalResults"])) # 최신 뉴스 건수 출력

for i in response_json['articles']:
    print("author = " + str(i['author'])) # 뉴스 건당 author 출력

