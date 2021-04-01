import requests

payload =  '["Veer-Zara is the best romantic movie i have seen . SRK and Preeti Zinta acting was great in it"]'
headers = {'Content-Type': 'application/json; format=pandas-records'}
request_uri = 'http://127.0.0.1:5000/invocations'

if __name__ == '__main__':
   try:
      response = requests.post(request_uri, data=payload, headers=headers)
      print(response.content)
   except Exception as ex:
      raise(ex)
