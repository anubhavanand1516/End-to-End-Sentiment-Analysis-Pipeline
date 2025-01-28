import requests

url = "http://127.0.0.1:5000/predict"
data = {"review_text": "This movie was absolutely amazing!"}

response = requests.post(url, json=data)
print(response.json())
