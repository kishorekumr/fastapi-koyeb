import subprocess
import sys
from typing import Union
from typing import List, Dict, Any
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import hashlib
from scrapingbee import ScrapingBeeClient
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from fastapi.responses import Response
from fastapi import Request
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi import Query
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
import hashlib
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests


from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://kite.zerodha.com","https://api.kite.trade"],  # Specify allowed origins
    allow_origins=["*"],
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

class HistoricalData(BaseModel):
    HOYear: str
    HOOpen: float
    HOHigh: float
    HOLow: float
    HOClose: float
    HOVolume: int
    HOTurnover: int

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

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("favicon.ico")

class KiteLoginRequest(BaseModel):
    account_username: str
    account_password: str
    account_two_fa: str
    api_key: str
    api_secret: str

def Kite_Login_api(account_username, account_password, account_two_fa, api_key, api_secret):
    reqSession = requests.Session()
    loginurl = "https://kite.zerodha.com/api/login"
    twofaUrl = "https://kite.zerodha.com/api/twofa"
    if len(account_two_fa)>6:
        account_two_fa=TOTP(account_two_fa).now()
        account_two_fa=account_two_fa.zfill(6)
    # Step 1: login and get request_id
    resp1 = reqSession.post(
        loginurl,
        data={"user_id": account_username, "password": account_password},
        timeout=10
    )
    resp1.raise_for_status()
    request_id = resp1.json()["data"]["request_id"]

    # Step 2: submit 2FA
    resp2 = reqSession.post(
        twofaUrl,
        data={
            "user_id": account_username,
            "request_id": request_id,
            "twofa_value": account_two_fa,
            "twofa_type": "totp",
            "skip_session": "true"
        },
        timeout=10
    )
    resp2.raise_for_status()

    # Step 3: capture the redirect URL and extract request_token
    api_session = reqSession.get(f"https://kite.trade/connect/login?api_key={api_key}", timeout=10)
    try:
        request_token = api_session.url.split("request_token=")[1].split("&")[0]
    except Exception:
        raise Exception("Could not extract request_token from redirect URL")

    # Step 4: generate checksum & exchange for access_token
    checksum = hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode("utf-8")).hexdigest()
    token_resp = reqSession.post(
        "https://api.kite.trade/session/token",
        data={
            "api_key": api_key,
            "request_token": request_token,
            "checksum": checksum
        },
        timeout=10
    )
    token_resp.raise_for_status()
    data = token_resp.json()
    if data.get("data", {}).get("access_token"):
        return data["data"]["access_token"]
    raise Exception("Failed to obtain access_token")

@app.post("/kite_login")
def kite_login(req: KiteLoginRequest):
    """
    Authenticate with Kite and return an access_token.
    """
    try:
        token = Kite_Login_api(
            req.account_username,
            req.account_password,
            req.account_two_fa,
            req.api_key,
            req.api_secret
        )
        return {"access_token": token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class KiteWebLoginRequest(BaseModel):
    account_username: str
    account_password: str
    account_two_fa: str

def kite_web_login_api(kite_username: str, kite_password: str, kite_pin: str) -> str:
    try:
        session = requests.Session()

        r = session.post(
            'https://kite.zerodha.com/api/login',
            data={
                'user_id': kite_username,
                'password': kite_password
            },
            timeout=10
        )
        r.raise_for_status()
        request_id = r.json()['data']['request_id']

        r2 = session.post(
            'https://kite.zerodha.com/api/twofa',
            data={
                'request_id': request_id,
                'twofa_value': kite_pin,
                'user_id': kite_username
            },
            timeout=10
        )
        r2.raise_for_status()

        cookies_dict = r2.cookies.get_dict()
        if 'enctoken' not in cookies_dict:
            raise Exception("enctoken not found in cookies")

        return cookies_dict['enctoken']

    except Exception as e:
        raise Exception(f"Kite web login failed: {str(e)}")

@app.post("/kite_web_login")
def kite_web_login(req: KiteWebLoginRequest):
    """
    Login to Kite without API Key/Secret and return the enctoken.
    """
    try:
        enctoken = kite_web_login_api(req.account_username, req.account_password, req.account_two_fa)
        return {"enctoken": enctoken}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route(
    "/kite_api_proxy/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
)
async def kite_proxy(request: Request, full_path: str):
    # 1) Build the real Kite URL
    target_url = f"https://api.kite.trade/{full_path}"

    # 2) Copy through only the headers Kite cares about
    forwarded_headers = {
        # preserve auth so “token APIKEY:ACCESSTOKEN” still works
        "Authorization": request.headers.get("authorization", ""),
        # content-type for POST bodies
        "Content-Type": request.headers.get("content-type", ""),
    }

    # 3) Forward query-params and body
    async with httpx.AsyncClient(timeout=10) as client:
        proxy_resp = await client.request(
            request.method,
            target_url,
            headers=forwarded_headers,
            params=request.query_params,
            content=await request.body()
        )

    # 4) Relay Kite’s response (status, headers, body) back to the frontend
    return Response(
        content=proxy_resp.content,
        status_code=proxy_resp.status_code,
        headers={
            # You can whitelist headers here if you like:
            k: v for k, v in proxy_resp.headers.items()
            if k.lower() in ("content-type", "x-kite-version", "x-rate-limit-limit")
        }
    )


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



@app.get("/history_it2/{exch}/{fincode}" ,response_model=Union[List[Dict[str, Any]], Dict[str, Any]])
def get_it_history(exch: str, fincode: str):
    try:
        url=f'https://www.indiratrade.com/Ajaxpages/companyprofile/CompanyHistoricalVol.aspx?Option={exch}&FinCode={fincode}&fmonth=OCT&fyear=2024&lmonth=NOV&lyear=2024&pageNo=1&PageSize=50'
        print(url)
        client = ScrapingBeeClient(api_key='3YJNM84LOHW51BJ8WMRLMNERNQ4F5U9CLVVNXQ8MUJZA4LL2IWRBIS48QK3PD3WXKQRTS8OIWQ32CJE8')
        response = client.get(url)
        if response.status_code == 200:
            print("Response Data:", response.text)
            data = response.json()
            processed_data = []
            for item in data:
                processed_data.append({
                    'time': item.get('HOYear'),
                    'open': float(item.get('HOOpen', 0)),  # Add default for safety
                    'high': float(item.get('HOHigh', 0)),
                    'low': float(item.get('HOLow', 0)),
                    'close': float(item.get('HOClose', 0)),
                    'volume': int(item.get('HOVolume', 0)),
                })
        print(processed_data)
        return processed_data
    except:
        return {"Message":"Error"}

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


@app.get("/shoonya_web/{user_id}/{password}/{totp}", response_class=PlainTextResponse)
def get_shoonya_web(user_id: str,password: str,totp: str):
    url='https://trade.shoonya.com/NorenWClientWeb/QuickAuth'
    headers = {'Content-Type': 'application/json'}
    pwd = hashlib.sha256(unquote(password).encode('utf-8')).hexdigest()
    app_key='fe704c430d6d69d6b0dbda9f106e0a128892143da06dad96a7c1642e66531ee1'
    print(url)
    
    try:
        if len(totp)>6:
            totp=TOTP(totp).now()
            totp=totp.zfill(6)

        payload = f'jData={{"uid":{user_id},"pwd":"{pwd}","factor2":"{totp}","apkversion":"20240711","imei":"0aafd02e-cd3d-4191-8635-b3121d126654","vc":"NOREN_WEB","appkey":"{app_key}","source":"WEB","addldivinf":"Chrome-131.0.0.0"}}'
        response = requests.post('https://trade.shoonya.com/NorenWClientWeb/QuickAuth', data=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        # Return the token from the response
        susertoken = response_data.get('susertoken')
        if not susertoken:
            raise HTTPException(status_code=400, detail=f"Token not found. Full response: {response_data}")
        return susertoken
    except requests.exceptions.Timeout:
        # Handle timeout exception
        raise HTTPException(status_code=504, detail="Request to Shoonya API timed out.")
    except requests.exceptions.RequestException as e:
        # Handle all other request-related exceptions
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except KeyError:
        # Handle cases where the expected key is not in the response JSON
        raise HTTPException(status_code=500, detail="Response does not contain the expected key 'susertoken'")
    except Exception as ex:
        return str(ex)

@app.get("/shoonya_webs3/{user_id}/{password}/{totp}", response_class=PlainTextResponse)
def get_shoonya_web(user_id: str, password: str, totp: str):
    url = 'https://trade.shoonya.com/NorenWClientWebS3/QuickAuth'
    headers = {'Content-Type': 'text/plain; charset=utf-8'}
    pwd = hashlib.sha256(unquote(password).encode('utf-8')).hexdigest()
    appkey="b050537388ccd14a20c7a132661303451f14a83e6814133edff72ce71cb05ee4"
    print(user_id,password,totp)
    try:
        if len(totp) > 6:
            totp = TOTP(totp).now()
            totp = totp.zfill(6)
        else:
            print(totp)
        payload = f'jData={{"uid":"{user_id}","pwd":"{pwd}","factor2":"{totp}","apkversion":"1.0.0","imei":"abc123","vc":"SHOONYA_WEB","appkey":"{appkey}","source":"WEB"}}'

        response = requests.post(url, data=payload, headers=headers, timeout=10)
        
        response.raise_for_status()
        response_data = response.json()
        susertoken = response_data.get('susertoken')
        if not susertoken:
            raise HTTPException(status_code=400, detail=f"Token not found. Full response: {response_data}")
        return susertoken
    except requests.exceptions.Timeout:
        # Handle timeout exception
        raise HTTPException(status_code=504, detail="Request to Shoonya API timed out.")
    except requests.exceptions.RequestException as e:
        # Handle all other request-related exceptions
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except KeyError:
        # Handle cases where the expected key is not in the response JSON
        raise HTTPException(status_code=500, detail="Response does not contain the expected key 'susertoken'")

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
        base_url = "https://www.nseindia.com/"
        api_url  = base_url + "api/holiday-master?type=trading"

        # 1) browser-like headers for *all* requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;"
                      "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": base_url,
            "Connection": "keep-alive"
        }

        with requests.Session() as session:
            # 2) hit the home page to get cookies + inline JS
            r_home = session.get(base_url, headers=headers, timeout=10)
            r_home.raise_for_status()

            # 3) scrape out the nseappid token from the inline scripts
            #    It usually lives in a JS object like: "nseappid":"<TOKEN>"
            text = r_home.text
            m = re.search(r'"nseappid"\s*:\s*"([^"]+)"', text)
            if not m:
                raise RuntimeError("Could not find nseappid in homepage HTML")
            token = m.group(1)

            # 4) add the token header and some AJAX headers
            api_headers = headers.copy()
            api_headers.update({
                "Accept": "application/json, text/plain, */*",
                "X-Requested-With": "XMLHttpRequest",
                "nseappid": token,
            })

            # 5) finally call the holiday API
            r_api = session.get(api_url, headers=api_headers, timeout=10)
            r_api.raise_for_status()

            data = r_api.json()

        # (Optional) quick check if today is a holiday
        today = datetime.today().date().isoformat()
        trading_dates = [ d["tradingDate"] for d in data.get("CM", []) ]
        is_holiday = today not in trading_dates

        return {
            "today": today,
            "isHoliday": is_holiday,
            "holidays": data["CM"]
        }

    except Exception as e:
        # FastAPI will return a 500 if you re-raise, or you can
        # explicitly return a 400/500 yourself.
        raise HTTPException(status_code=500, detail=str(e))


def _fetch_holiday_data():
    base_url   = "https://www.nseindia.com"
    landing    = base_url + "/market-data"                         # ← not just “/”
    api_url    = base_url + "/api/holiday-master?type=trading"

    # Fully Chrome-like headers
    common_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;"
                  "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_url,
        "Connection": "keep-alive",
        # Chrome 115+ typical fetch client hints
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
    }

    with requests.Session() as session:
        # 1) land on /market-data to get cookies + inline JS
        r1 = session.get(landing, headers=common_headers, timeout=10)
        r1.raise_for_status()

        # 2) extract the nseappid token
        match = re.search(r'"nseappid"\s*:\s*"([^"]+)"', r1.text)
        if not match:
            raise RuntimeError("nseappid token not found in HTML")
        token = match.group(1)

        # 3) prepare API headers (AJAX style)
        api_headers = common_headers.copy()
        api_headers.update({
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "nseappid": token
        })

        # 4) finally call the holidays API
        r2 = session.get(api_url, headers=api_headers, timeout=10)
        r2.raise_for_status()
        return r2.json()

@app.get("/holidays/all")
def list_all_holidays():
    try:
        payload = _fetch_holiday_data()
        cm      = payload.get("CM", [])
        return {
            "holidays": [
                {"date":  h["tradingDate"], "purpose": h.get("purpose","").strip()}
                for h in cm
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _make_nse_session():
    """
    1) start a session
    2) land on /market-data to pick up cookies + JS blob
    3) extract the nseappid token
    returns: (session, token, common_headers)
    """
    landing_url = "https://www.nseindia.com/market-data"
    common_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;"
                  "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        # Chrome-style fetch headers
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
    }

    session = requests.Session()
    resp = session.get(landing_url, headers=common_headers, timeout=10)
    resp.raise_for_status()

    m = re.search(r'"nseappid"\s*:\s*"([^"]+)"', resp.text)
    if not m:
        raise RuntimeError("Unable to find nseappid in landing HTML")
    token = m.group(1)

    return session, token, common_headers


@app.get("/stock-indices")
def get_stock_indices(
    index: str = Query(
        "NIFTY 50",
        title="Index name",
        description="Name of the index, e.g. 'NIFTY 50' or 'NIFTY TOTAL MARKET'"
    )
):
    """
    Fetches and returns the JSON from:
      https://www.nseindia.com/api/equity-stockIndices?index=<index>
    """

    try:
        session, token, headers = _make_nse_session()

        # build the API URL
        url = (
            "https://www.nseindia.com/api/equity-stockIndices"
            "?index=" + urllib.parse.quote_plus(index)
        )

        # upgrade headers for AJAX + token
        api_headers = headers.copy()
        api_headers.update({
            "Accept": "*/*",                   # like the browser does
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "nseappid": token,
        })

        resp = session.get(url, headers=api_headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    except Exception as e:
        # bubble up as a 500 with the error message
        raise HTTPException(status_code=500, detail=str(e))

import subprocess

@app.post("/install-package")
def install_package(package_name: str):
    """
    Installs a Python package using pip via subprocess.
    Example: /install-package?package_name=selenium
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            check=True
        )
        return {"message": f"Successfully installed '{package_name}'", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to install '{package_name}'", "details": e.stderr}

# if __name__ == '__main__':
#     try:
#         import selenium
#         print("Selenium is installed and working.")
#     except ImportError:
#         print("selenium not found. Installing...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium"])
#         print("selenium installed successfully.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
    
