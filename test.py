from preprocessMethod import preprocessData
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import os

with open("D:/360MoveData/Users/31365/Desktop/dataset/1.txt", "r", encoding='utf-8') as file:
    file = file.read().lower().split()
    str1 = preprocessData(file)
    print (str1)