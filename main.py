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
from fastapi import Body
from typing import Any
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
import traceback
import json
import urllib.parse
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
class QuickAuthReq(BaseModel):
    userid:      str
    password:    str
    totp_secret: str
    api_key:     str
    imei:        str
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

@app.get("/koyeb_ping")
async def ping_url(url: str = Query(..., description="Target URL to ping")):
    if not url.startswith("http://") and not url.startswith("https://"):
        return {"url": url, "error": "URL must start with http:// or https://"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), verify=False) as client:
            resp = await client.get(url)

        # Check if response is non-200
        if resp.status_code != 200:
            return {
                "url": url,
                "status_code": resp.status_code,
                "error": f"Non-200 status: {resp.status_code}",
                "preview": resp.text[:200]
            }

        return {
            "url": url,
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "preview": resp.text[:200]
        }

    except httpx.ConnectTimeout:
        return {"url": url, "error": "Connect timeout"}
    except httpx.RequestError as e:
        return {"url": url, "error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"url": url, "error": f"Unexpected error: {str(e)}"}
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




 # or use symphony.acagarwal.com:3000 if needed

@app.api_route("/motilal_api_proxy/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def xts_proxy(request: Request, full_path: str):
    XTS_BASE_URL = "https://moxtsapi.motilaloswal.com:3000" 
    target_url = f"{XTS_BASE_URL}/{full_path}"

    # 2. Forward only necessary headers
    forwarded_headers = {
        "Authorization": request.headers.get("authorization", ""),
        "Content-Type": request.headers.get("content-type", ""),
        "userID": request.headers.get("userid", ""),
    }

    # 3. Handle login endpoint: /user/session → add source=WEBAPI
    body = await request.body()
    if full_path.lower().endswith("user/session") and request.method == "POST":
        try:
            data = json.loads(body.decode())
            data["source"] = "WEBAPI"
            body = json.dumps(data).encode()
        except Exception as e:
            return Response(content=f"❌ Failed to parse or modify login body: {e}", status_code=400)

    # 4. Forward the request
    async with httpx.AsyncClient(timeout=15, verify=False) as client:
        proxy_resp = await client.request(
            method=request.method,
            url=target_url,
            headers=forwarded_headers,
            params=request.query_params,
            content=body
        )

    # 5. Relay response
    return Response(
        content=proxy_resp.content,
        status_code=proxy_resp.status_code,
        headers={
            k: v for k, v in proxy_resp.headers.items()
            if k.lower() in ("content-type", "x-xts-version")
        }
    )


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

@app.get("/shoonya_api/{user_id}/{password}/{totp}/{api_key}/{imei}", response_class=PlainTextResponse)
def get_shoonya_web(user_id: str, password: str, totp: str, api_key: str, imei: str):
    url = 'https://api.shoonya.com/NorenWClientTP/QuickAuth'
    headers = {'Content-Type': 'text/plain; charset=utf-8'}
    pwd = hashlib.sha256(password.encode('utf-8')).hexdigest()
    app_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    print(user_id,password,totp)
    try:
        if len(totp) > 6:
            totp = TOTP(totp).now()
            totp = totp.zfill(6)
        else:
            print(totp)
        payload = f'jData={{"uid":"{user_id}","pwd":"{pwd}","factor2":"{totp}","apkversion":"1.0.0","imei":"abc123","vc":"{user_id}_U","appkey":"{app_key_hash}","source":"API"}}'

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
    
