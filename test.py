import requests

response = requests.post("http://127.0.0.1:5000/process", json={"text": "king, woman"})
print(response.json())
