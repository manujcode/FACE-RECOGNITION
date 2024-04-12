import warnings

warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import Series,DataFrame
url ="https://www.geeksforgeeks.org/problems/minimum-sum4058/1"

result = requests.get(url)
soup = BeautifulSoup(result.content,'lxml')
print(soup)

