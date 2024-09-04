from typing import Union
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"message": "Hello from FastAPI!"}
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
# import httpx
import requests
import uvicorn
from datetime import datetime, timedelta
import pytz
from pyotp import TOTP
# import pyotp
app = FastAPI()
import json
import urllib3
from urllib.parse import unquote
import base64
# url="https://api.investing.com/api/financialdata/historical/1195383?start-date=2023-10-12&end-date=2024-08-03&time-frame=Daily&add-missing-rows=false"
# url='https://api.investing.com/api/financialdata/historical/1195383?start-date=2024-08-02&end-date=2024-08-03&time-frame=Daily&add-missing-rows=false'
# headers = {
#     'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0',
#     'X-Requested-With': 'XMLHttpRequest',
#     'Referer': 'https://in.investing.com/',
#     'domain-id': 'in'}
# data = requests.get(url, headers=headers).json()
# print(data)
# # print(type(data)) # dict
# df=pd.DataFrame(data['data'])

import pandas as pd
######### Store in SQL token and ltp from websocket

# from apscheduler.schedulers.background import BackgroundScheduler

# symbol_to_token = None
# df = pd.read_csv('/home/kishorekumar/fastapi/nse_token.csv')
# symbol_to_token = dict(zip(df['Symbol'], df['Token']))
# @app.on_event("startup")
# def start_scheduler():
#     scheduler = BackgroundScheduler()
#     scheduler.add_job(read_csv_daily, 'cron', hour=8, minute=10)
#     scheduler.start()

# def read_csv_daily():
#     global symbol_to_token
#     try:
#         # csv_data = pd.read_csv('/home/kishorekumar/fastapi/nse_token.csv')
#         df = pd.read_csv('nse_token.csv')
#         symbol_to_token = dict(zip(df['Symbol'], df['Token']))
#         print("CSV data loaded successfully.")
#     except Exception as e:
#         print(f"Failed to read CSV file: {e}")


@app.get("/")
def read_root():
    #token = symbol_to_token["TCS"]
    return {"message": f"Hello from FastAPI (Ver-holidays) at {datetime.utcnow()}"}


@app.get("/quote-derivative")
def get_quote_derivative(symbol: str):
    url = f'https://www.nseindia.com/api/quote-derivative?symbol={symbol}'
    print(url)
    nseappid='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTcyMzExODI4MCwiZXhwIjoxNzIzMTI1NDgwfQ.xIprKwUtrSyAUF8VlNm454Q-XBYK0bt0YlfzDn6kngg'
    nseappid='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTcyMzE0NjAwNSwiZXhwIjoxNzIzMTUzMjA1fQ.rWOMzhQ2KQAaMcNOjIpRxSgF3GoZyg2LcZCZAnrIhrE'
    headers = {
        'cookie': f'nsit=A; nseappid={nseappid};'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(response)
        return response.json()
        # response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")
        return str(e)
    
    


@app.get("/holidays")
def get_holidays():
    try:
        print("holidays endpoint called")
        base_url = "https://www.nseindia.com/"
        url='https://www.nseindia.com/api/holiday-master?type=trading'
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                                 'like Gecko) '
                                 'Chrome/80.0.3987.149 Safari/537.36',
                   'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
            'Accept-Language': 'en,gu;q=0.9,hi;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'nseappid':'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTcyMjkyOTkwMCwiZXhwIjoxNzIyOTM3MTAwfQ.w1YSS7jf3Nn5KJfuQdYtbUBjDon2uYwMgLSRpg_Vi5k'
        }
        print(headers)
        with requests.Session() as session:
            response1=session.get(base_url, headers=headers, timeout=10)  # Establish session
            print(dict(response1.cookies))
            response2 = session.get(url, headers=headers,timeout=10)  # Fetch data
            
            if response2.ok:
                data = response2.json()
                print(data)
            else:
                print(f"Error: {response2.status_code}")
        json_data=response2.json()["CM"]
        today=datetime.today().date
        print(today)
        if today in holiday_df['tradingDate']:
            print("Holiday")
        else:
            print("Trading Day")
        response2.raise_for_status()
        return response2.json()
    except Exception as ex:
        return str(ex)
    # async with httpx.AsyncClient() as client:
    #     try:
    #         response = await client.get(url)
    #         response.raise_for_status()  # Raises HTTPError for bad responses
    #         data = response.json()
    #         return data
    #     except httpx.RequestError as e:
    #         raise HTTPException(status_code=500, detail=f"Request failed: {e}")
    #     except httpx.HTTPStatusError as e:
    #         raise HTTPException(status_code=response.status_code, detail=f"HTTP error: {e}")
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")








@app.get("/lic_check/{text}", response_class=PlainTextResponse)
def get_lic(text: str):
    # return str
    try:
        lic_list=['0191bb5c-cf09-7ecd-9f3c-fd2df1171511']
        if text in lic_list:
            return  "Valid"
        else:
            return "Lic Error
    except:
        return "Server Error"


@app.get("/totp/{text}", response_class=PlainTextResponse)
def get_text(text: str):
    # return str
    try:
        totp=TOTP(text).now()
        totp = totp.zfill(6)
        # return int(totp)
        return  totp
    except:
        return "Error"

@app.get("/totp_next/{text}", response_class=PlainTextResponse)
def get_text_next(text: str):
    # return str
    try:
        totp=TOTP(text).at(datetime.utcnow(),1)
        totp = totp.zfill(6)
        # return int(totp)
        return  totp
    except:
        return "Error"


@app.get("/kite_api/{uid}/{pwd}/{totp}", response_class=PlainTextResponse)
def get_kite_api(uid: str,pwd: str,totp: str):
    # return str
    try:
        uid=unquote(uid)
        if len(totp)>6:
            totp_now=TOTP(totp).now()
            totp_now = totp_now.zfill(6)
        else:
            totp_now=totp.zfill(6)
        # reqSession = urllib3.PoolManager() #requests.Session()
        reqSession =requests.Session()
        loginurl = "https://kite.zerodha.com/api/login"
        twofaUrl = "https://kite.zerodha.com/api/twofa"
        request_id = reqSession.request('POST',
            loginurl,
            data={
                "user_id": uid,
                "password": pwd,
            },
        ).json()["data"]["request_id"]
        reqSession.post(
            twofaUrl,
            data={"user_id": uid, "request_id": request_id, "twofa_value": totp_now},
        )
        API_Session = reqSession.request('GET',"https://kite.trade/connect/login?api_key=coinios")
        print(API_Session.url)
        API_Session = API_Session.url.split("#")[1]
        data=base64.b64decode(API_Session)
        data=json.loads(data)
        print(data)
        access_token = data["access_token"]
        return access_token
    except:
        return "Error"

@app.get("/kite_api2/{uid}/{pwd}/{totp}", response_class=PlainTextResponse)
def get_kite_api2(uid: str,pwd: str,totp: str):
    # return str
    try:
        uid=unquote(uid)
        # totp_now=TOTP(totp).now()
        # totp_now = totp_now.zfill(6)
        # reqSession = urllib3.PoolManager() #requests.Session()
        reqSession =requests.Session()
        loginurl = "https://kite.zerodha.com/api/login"
        twofaUrl = "https://kite.zerodha.com/api/twofa"
        request_id = reqSession.request('POST',
            loginurl,
            data={
                "user_id": uid,
                "password": pwd,
            },
        ).json()["data"]["request_id"]
        reqSession.post(
            twofaUrl,
            data={"user_id": uid, "request_id": request_id, "twofa_value": totp},
        )
        API_Session = reqSession.request('GET',"https://kite.trade/connect/login?api_key=coinios")
        print(API_Session.url)
        API_Session = API_Session.url.split("#")[1]
        data=base64.b64decode(API_Session)
        data=json.loads(data)
        print(data)
        access_token = data["access_token"]
        return access_token
    except:
        return "Error"

# @app.get("/nse_token/{text}", response_class=PlainTextResponse)
# def get_nse_token(text: str):
#     #global symbol_to_token
#     try:
#         token='11536'
#         #token = symbol_to_token["TCS"]
#         #token = symbol_to_token['3MINDIA']
#         #print(token)
#         # return int(totp)
#         #send_message(5618402434, text)
#         token = symbol_to_token[text]
#         return  str(token)
#     except:
#         return "Error"


@app.get("/investing/{inv_id}/{end_date}", response_class=PlainTextResponse)
def get_investing(inv_id: int,end_date:str):
    #global symbol_to_token
    try:
        http = urllib3.PoolManager()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - timedelta(days=1)

        # Format dates for URL
        start_date_str = start_date_obj.strftime("%Y-%m-%d")
        end_date_str = end_date_obj.strftime("%Y-%m-%d")

        url = "https://www.investing.com/indices/cnx-nifty-junior-historical-data"
        url = "https://es.investing.com/indices/cnx-nifty-junior-historical-data"
        
        # 0   02.08.2024  73.328,15  73.621,40  74.241,25  73.173,45  723,63M  -1.30%
        # r = requests.get(url)
        # html_content = StringIO(r.text)
        # print(html_content)
        r = http.request('GET', url)
        data = r.data.decode('utf-8')
        data_str = StringIO(data)
        # print(tables)
        tables = pd.read_html(data_str, decimal=',', thousands="'")
        # for i, table in enumerate(tables):
        #     print(f"\nTable {i}:\n", table.head())
        historical_data = tables[0]
        print(type(historical_data))
        print(historical_data)
        try:
            print(historical_data['Fecha'].iloc[0])
            return historical_data['Último'].iloc[0]
        except:
            print(historical_data['Date'].iloc[0])
            return historical_data['Price'].iloc[0]
        
        # # Construct URL
        # url = f"https://api.investing.com/api/financialdata/historical/1195383?start-date={start_date_str}&end-date={end_date_str}&time-frame=Daily&add-missing-rows=false"
        # # return url
        # headers = {
        #     'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0',
        #     'X-Requested-With': 'XMLHttpRequest',
        #     'Referer': 'https://www.investing.com/',
        #     'domain-id': 'www'
        # }
        # response = http.request('GET', url,headers=headers)
        # # response = urllib3.request("GET", url,headers=headers) #requests.get(url, headers=headers)
        # if response.status == 200:
        #     return response.json()
        # else:
        #     return response.status
        # if response.status_code == 200:
        #     data = response.json()
        #     df = pd.DataFrame(data['data'])

        #     # Ensure DataFrame is not empty
        #     if not df.empty:
        #         # Find the entry for the end_date
        #         df['rowDate'] = pd.to_datetime(df['rowDate'])
        #         end_date_entry = df[df['rowDate'] == end_date_obj]

        #         if not end_date_entry.empty:
        #             last_close = end_date_entry['last_close'].values[0]
        #             return {"inv_id": inv_id, "end_date": end_date_str, "last_close": last_close}
        #         else:
        #             return {"inv_id": inv_id, "end_date": end_date_str, "last_close": "No data available for the specified end_date"}
        #     else:
        #         return {"inv_id": inv_id, "end_date": end_date_str, "last_close": "No data available"}
        # else:
        #     raise HTTPException(status_code=response.status_code, detail="Request failed")

    except Exception as e:
        return {"error": str(e)}




# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)
# uvicorn.run(app, host="YOUR_HOST", port=YOUR_PORT, timeout_keep_alive=YOUR_TIMEOUT_IN_SECONDS)
    
