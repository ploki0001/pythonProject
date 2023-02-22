import os
from preprocessMethod import preprocessData
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


filenames=os.listdir(r'D:/360MoveData/Users/31365/Desktop/dataset')
count = 1
index = []
train = []
test = []
trainlable = []
testlable = []
for i in filenames:
    index.append(i)
    path = 'D:/360MoveData/Users/31365/Desktop/dataset/'+i
    count += 1
    with open(path, "r", encoding='utf-8') as file:
        str1 = ""
        str2 = ""
        a = preprocessData(file)
        b = a[:round(len(a) / 2)]
        c = a[round(len(a) / 2):]
        for ele in b:
            str1 += ele + ' '
        for ele in c:
            str2 += ele + ' '
        train.append(str1)
        trainlable.append(i)
        test.append(str2)
        testlable.append(i)



tfidfvectorizer = TfidfVectorizer(max_df=0.5)

traindata = tfidfvectorizer.fit_transform(train)
testdata = tfidfvectorizer.transform(test)


nb_model = MultinomialNB(alpha=0.001)
nb_model.fit(traindata,trainlable)
predict_test = nb_model.predict(testdata)
print("多项式朴素贝叶斯文本分类的准确率为：",metrics.accuracy_score(predict_test,testlable))






