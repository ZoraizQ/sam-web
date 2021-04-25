import requests
import json
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("args error")
    exit(1)

url = sys.argv[1]
api = "/predict"
url += api

img_path = 'test.jpg'
img = {'media': open(img_path, 'rb')}

r = requests.post(url, files=img)
print(r.ok)
print(r.status_code)
print(r.headers)
print(r.encoding)


if r.status_code == 200:
    with open("p_"+img_path, 'wb') as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
