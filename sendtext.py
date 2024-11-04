import requests

url = 'http://127.0.0.1:5000/analyze'
data = {
    'text1': 'This is the first sentence.',
    'text2': 'This was the last sentence.'
}

response = requests.post(url, json=data)
print(response.json())