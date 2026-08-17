import requests
import json

# Test health endpoint
resp = requests.get('http://localhost:8000/health')
print(f'Health: {resp.json()}')

# Test API health
resp = requests.get('http://localhost:8000/api/v1/health')
print(f'API Health: {resp.json()}')

# List datasets
resp = requests.get('http://localhost:8000/api/v1/datasets/')
print(f'List datasets: {resp.json()}')