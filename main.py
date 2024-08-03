from typing import Union
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
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
# import requests
import uvicorn
from datetime import datetime, timedelta
import pytz
from pyotp import TOTP
# import pyotp
app = FastAPI()
import json
import urllib3


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
    return {"message": f"Hello from FastAPI at {datetime.utcnow()}"}

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


@app.get("/investing/{inv_id}/{end_date}")
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
        r = requests.get(url)
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

    
