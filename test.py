import os
import linecache
from preprocessMethod import preprocessData
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


filenames=os.listdir(r'D:\360MoveData\Users\31365\Desktop\新建文件夹')
train = []
test = []
trainlable = []
testlable = []
for i in filenames:
    path = 'D:/360MoveData/Users/31365/Desktop/新建文件夹/'+i
    with open(path, "r", encoding='utf-8') as file:
        count = 1
        lines = len(file.readlines())
        trainLines = round(lines*0.5)
        for count in range(lines+1):
            if count <= trainLines:
                a = preprocessData(linecache.getline(path, count))
                print (a)
            else:
                b = preprocessData(linecache.getline(path, count))













