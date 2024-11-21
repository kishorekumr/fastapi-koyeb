from typing import Union
from typing import List, Dict, Any
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests

from fastapi import Request
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import httpx
import requests
import uvicorn
from datetime import datetime, timedelta, timezone
import pytz
from pyotp import TOTP
# import pyotp
app = FastAPI()
import json
import time
import urllib3
from urllib.parse import unquote, quote
import base64

from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kite.zerodha.com"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: Union[str, None] = None

class DataFrameRow(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: int

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    return obj

def convert_to_iso8601(excel_time: float):
    # Convert Excel float time to datetime object
    if excel_time >= 59:
        excel_time -= 2
    excel_epoch = datetime(1900, 1, 1)
    timestamp = excel_epoch + timedelta(days=excel_time)
    return timestamp.strftime('%Y-%m-%dT%H:%M:%S.000Z')


@app.get("/")
def read_root():
    #token = symbol_to_token["TCS"]
    return {"message": f"Hello from FastAPI (Ver-mchistory) at {datetime.utcnow()}"}

@app.head("/")
def read_root_head():
    return "OK"

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
    
    
# https://priceapi.moneycontrol.com/techCharts/indianMarket/stock/history?symbol=

@app.get("/history/{symbol}",response_model=List[DataFrameRow]) #,response_model=List[Dict[str, Any]]
def get_mc_history(symbol: str):

    if ":" in symbol:
        sym_quote=symbol.replace(":","%3B")
        url = f'https://priceapi.moneycontrol.com/techCharts/indianMarket/index/history?symbol={sym_quote}'
    else:
        url = f'https://priceapi.moneycontrol.com/techCharts/indianMarket/stock/history?symbol={symbol}'
    
    print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    today = datetime.now()
    thirty_days_ago = today - timedelta(days=30)

    # Convert to UNIX timestamps
    to_timestamp = int(time.mktime(today.timetuple()))
    from_timestamp = int(time.mktime(thirty_days_ago.timetuple()))

    # Format the result as needed
    result = f"&resolution=1D&from={from_timestamp}&to={to_timestamp}&countback=30"
    url=url+result
    # return url
    print(url)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        # Print the JSON response to understand its structure
        print("JSON Data:", data)
        data = {k: [convert_to_serializable(i) for i in v] if isinstance(v, list) else convert_to_serializable(v) for k, v in data.items()}

        if data.get('s') == 'ok':
            # Convert the data to a DataFrame
            df = pd.DataFrame({
                'time': [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d') for t in data['t']],
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            json_compatible_df = jsonable_encoder(df.to_dict(orient='records'))
            return JSONResponse(content=json_compatible_df)
            return df.to_string()
            return df.to_dict(orient='records')
        else:
            return pd.DataFrame() 
        # Convert the JSON response to a DataFrame (assuming it's a list of records)
        # if isinstance(json_data, list):  # If it's a list of dictionaries
        #     return pd.DataFrame(json_data)
        # elif isinstance(json_data, dict):  # If it's a dictionary
        #     return pd.DataFrame([json_data])
        # else:
        #     print("Unexpected JSON structure.")
        #     return pd.DataFrame()
        # response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")
        return str(e)
    except Exception as ex:
        print(f"Error in symbol: {str(ex)}")


@app.get("/history_mcx/{symbol}",response_model=List[DataFrameRow]) #,response_model=List[Dict[str, Any]]
def get_mc_history(symbol: str):

    if ":" in symbol:
        sym_quote=symbol.replace(":","%3B")
        url = f'https://priceapi.moneycontrol.com/techCharts/commodity/history?symbol={sym_quote}'
    else:
        url = f'https://priceapi.moneycontrol.com/techCharts/commodity/history?symbol={symbol}'
    
    print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    today = datetime.now()
    thirty_days_ago = today - timedelta(days=60)

    # Convert to UNIX timestamps
    to_timestamp = int(time.mktime(today.timetuple()))
    from_timestamp = int(time.mktime(thirty_days_ago.timetuple()))

    # Format the result as needed
    result = f"&resolution=1D&from={from_timestamp}&to={to_timestamp}&countback=60"
    url=url+result
    # return url
    print(url)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        # Print the JSON response to understand its structure
        print("JSON Data:", data)
        data = {k: [convert_to_serializable(i) for i in v] if isinstance(v, list) else convert_to_serializable(v) for k, v in data.items()}

        if data.get('s') == 'ok':
            # Convert the data to a DataFrame
            df = pd.DataFrame({
                'time': [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d') for t in data['t']],
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                # 'volume': data['v']
            })
            json_compatible_df = jsonable_encoder(df.to_dict(orient='records'))
            return JSONResponse(content=json_compatible_df)
            return df.to_string()
            return df.to_dict(orient='records')
        else:
            return pd.DataFrame() 
        # Convert the JSON response to a DataFrame (assuming it's a list of records)
        # if isinstance(json_data, list):  # If it's a list of dictionaries
        #     return pd.DataFrame(json_data)
        # elif isinstance(json_data, dict):  # If it's a dictionary
        #     return pd.DataFrame([json_data])
        # else:
        #     print("Unexpected JSON structure.")
        #     return pd.DataFrame()
        # response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")
        return str(e)
    except Exception as ex:
        print(f"Error in symbol: {str(ex)}")


@app.get("/history_it/{fincode}",response_model=List[DataFrameRow]) #,response_model=List[Dict[str, Any]]
def get_mc_history(fincode: str):
    # get months
    url = f'https://www.indiratrade.com/Ajaxpages/companyprofile/CompanyHistoricalVol.aspx?Option=NSE&FinCode={fincode}'
    try:
        import requests
        from requests.packages.urllib3.contrib.pyopenssl import inject_into_urllib3
        inject_into_urllib3()
        print("requests[security] is installed and working.")
    except ImportError:
        print("requests[security] not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests[security]"])
        print("requests[security] installed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")    
    print(url)
    headers1 = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
    }
    today = datetime.now()
    current_month = today.strftime('%b').upper()  # Short month name (e.g., 'SEP')
    current_year = today.year

    # Previous month and year
    prev_month = (today.replace(day=1) - timedelta(days=1)).strftime('%b').upper()
    prev_year = (today.replace(day=1) - timedelta(days=1)).year

    # Format the result
    result = f'&fmonth={prev_month}&fyear={prev_year}&lmonth={current_month}&lyear={current_year}&pageNo=1&PageSize=50'

    url=url+result
    # return url
    print(url)
    try:
        session = requests.Session()
        response = session.get("https://www.indiratrade.com/Ajaxpages/companyprofile/CompanyHistoricalVol.aspx", headers=headers1)
        print(response)
        print(response.text)
        print("Cookies:", session.cookies.get_dict())
        response = session.get(url, headers=headers)
        data = response.json()
        print(type(data))
        # Print the JSON response to understand its structure
        print("JSON Data:", data)
        processed_data = []
        for item in data:
            # Process each item from the list if needed
            processed_data.append({
                'time': item.get('HOYear'),
                'open': item.get('HOOpen'),
                'high': item.get('HOHigh'),
                'low': item.get('HOLow'),
                'close': item.get('HOClose'),
                'volume': item.get('HOVolume')
            })
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        json_compatible_df = jsonable_encoder(df.to_dict(orient='records'))
        return JSONResponse(content=json_compatible_df)
        # else:
            # return pd.DataFrame() 
    except Exception as ex:
        print(f"Error in symbol: {str(ex)}")
        return pd.DataFrame() 

@app.get("/smallcase/{sc_id}", response_class=PlainTextResponse)
async def smallcase(sc_id: str):
    url = f"https://api.smallcase.com/sam/smallcases?scid={sc_id}"  # URL with the sc_id parameter
    
    async with httpx.AsyncClient() as client:
        try:
            # Make the GET request
            response = await client.get(url)
            
            # Check for a successful response (status code 200)
            if response.status_code == 200:
                # Parse the JSON response
                json_response = response.json()
                
                # Access systemCalculatedMIA from the response
                system_calculated_mia = json_response['data']['stats']['systemCalculatedMIA']

                system_calculated_mia = str(system_calculated_mia).replace(",", "")
                
                # Return the value or log it
                return str(system_calculated_mia)
            else:
                # Raise HTTPException if response code is not 200
                raise HTTPException(status_code=response.status_code, detail="Error fetching data")
        
        except Exception as e:
            # Handle other exceptions
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch_ltp/{appkey}/{ses_token}/{stock_code}/{exch_code}/{from_date}/{to_date}/{interval}/{product_type}/{expiry_date}/{right}/{strike_price}", response_class=JSONResponse)
async def fetch_ltp(
    appkey: str,
    ses_token: str,
    stock_code: str,
    exch_code: str,
    from_date: str,
    to_date: str, 
    interval: str,
    product_type: str,
    expiry_date: str,
    right: str,
    strike_price: str
):
    # Decode any URL-encoded strings
    appkey = unquote(appkey)
    ses_token = unquote(ses_token)
    stock_code = unquote(stock_code)
    exch_code = unquote(exch_code)
    from_date = unquote(from_date)
    to_date = unquote(to_date)
    interval = unquote(interval)
    product_type = unquote(product_type)
    expiry_date = unquote(expiry_date)
    right = unquote(right)
    strike_price = unquote(strike_price)

    # Construct the URL and headers
    url = "https://breezeapi.icicidirect.com/api/v2/historicalcharts"  # Replace with actual API URL
    # from_date = to_date
    # Define the query parameters for the request
    params = {
        "stock_code": stock_code,
        "exch_code": exch_code,
        "from_date": from_date,
        "to_date": to_date,
        "interval": interval,
        "product_type": product_type,
        "expiry_date": expiry_date,
        "right": right,
        "strike_price": strike_price
    }
    headers = {
        'X-SessionToken': ses_token,
        'apikey': appkey
    }
    # return params,headers
    print(params,headers)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers,timeout=5)

            # Check if response is successful
            if response.status_code == 200:
                print(response.json())
                json_response = response.json()
                return json_response['Success']

        except Exception as e:
            print(f"Exception: {str(e)}")


@app.get("/fetch_ltp_excel/{appkey}/{ses_token}/{stock_code}/{exch_code}/{to_date}/{interval}/{product_type}/{expiry_date}/{right}/{strike_price}", response_class=PlainTextResponse)
async def fetch_ltp_excel(
    appkey: str,
    ses_token: str,
    stock_code: str,
    exch_code: str,
    to_date: float,
    interval: str,
    product_type: str,
    expiry_date: float,
    right: str,
    strike_price: float
):
    # Decode any URL-encoded strings
    appkey = unquote(appkey)
    ses_token = unquote(ses_token)
    stock_code = unquote(stock_code)
    exch_code = unquote(exch_code)
    # to_date = unquote(to_date)
    interval = unquote(interval)
    product_type = unquote(product_type)
    # expiry_date = unquote(expiry_date)
    right = unquote(right)
    # strike_price = unquote(strike_price)

    # Construct the URL and headers
    url = "https://breezeapi.icicidirect.com/api/v2/historicalcharts"  # Replace with actual API URL
    expiry_date = convert_to_iso8601(expiry_date)
    to_date = convert_to_iso8601(to_date)
    from_date = to_date
    # Define the query parameters for the request
    params = {
        "stock_code": stock_code,
        "exch_code": exch_code,
        "from_date": from_date,
        "to_date": to_date,
        "interval": interval,
        "product_type": product_type,
        "expiry_date": expiry_date,
        "right": right,
        "strike_price": strike_price
    }
    headers = {
        'X-SessionToken': ses_token,
        'apikey': appkey
    }
    # return params,headers
    # Make the GET request with httpx
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers,timeout=5)

            # Check if response is successful
            if response.status_code == 200:
                json_response = response.json()
                print(json_response)
                # return json_response['Status']
                return str(json_response['Success'][0]['close'])

        except Exception as ex:
            return str(json_response['Error'])
            print(f"Exception:{str(ex)}")



@app.get("/lic_check/{text}", response_class=PlainTextResponse)
def get_lic(text: str):
    # return str
    try:
        lic_list=['0191bb5c-cf09-7ecd-9f3c-fd2df1171511']
        if text in lic_list:
            return  "Valid"
        else:
            return "Lic Error"
    except:
        return "Server Error"

@app.post("/totp", response_class=JSONResponse)
async def get_totp(request: Request):
    try:
        # Parse the incoming JSON body
        body = await request.json()
        secret_request = body.get("secretRequest")

        # Generate the TOTP
        totp = TOTP(secret_request).now()
        totp = totp.zfill(6)
        
        # Return the TOTP as a string
        return JSONResponse(content={"token": totp})
    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/totp", response_class=JSONResponse)
async def get_totp(request: Request):
    try:
        # Parse the incoming JSON body
        body = await request.json()
        secret_request = body.get("secretRequest")

        if not secret_request:
            raise HTTPException(status_code=400, detail="Missing 'secretRequest' in request body.")

        # Generate the TOTP
        totp = TOTP(secret_request).now()
        totp = totp.zfill(6)
        
        # Return the TOTP as a string
        return JSONResponse(content={"token": totp})
    
    except HTTPException as http_exc:
        return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Error: {str(e)}"})

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





# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)
# uvicorn.run(app, host="YOUR_HOST", port=YOUR_PORT, timeout_keep_alive=YOUR_TIMEOUT_IN_SECONDS)
    
