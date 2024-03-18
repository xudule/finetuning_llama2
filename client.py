import requests
import json

url = "http://0.0.0.0:5000/chat"
headers = {'Content-Type': 'application/json'}

print("MyBuddy: Hey there! Ask me anything.")
while True:
    question = input("You: ")
    data = {'question': question}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("MyBuddy: ", response.json()['answer'])