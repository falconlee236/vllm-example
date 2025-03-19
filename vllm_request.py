import requests
import json

default_model_res = requests.get("http://localhost:8000/v1/models")
print(f"default_model_Res \n {json.dumps(default_model_res.json(), sort_keys=True, indent=4)}")

url = "http://localhost:8000/v1/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "./output/model",
    "prompt": "Who is a Elon Musk?",
    "max_tokens": 256,
    "temperature": 0
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(json.dumps(response.json(), sort_keys=True, indent=4))