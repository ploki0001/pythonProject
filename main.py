import os
import linecache
from preprocessMethod import preprocessData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn import metrics

filenames = os.listdir(r'D:/360MoveData/Users/31365/Desktop/dataset')
train = []
test = []
trainlabel = []
testlabel = []
labels=[]
for i in filenames:
    labels.append(i)
    path = 'D:/360MoveData/Users/31365/Desktop/dataset/' + i
    with open(path, "r", encoding='utf-8') as file:
        count = 1
        lines = len(file.readlines())
        trainLines = round(lines * 0.6)
        for count in range(lines + 1):
            if count <= trainLines:
                a = preprocessData(linecache.getline(path, count).strip())
                train.append(a)
                trainlabel.append(i)
            else:
                b = preprocessData(linecache.getline(path, count).strip())
                test.append(b)
                testlabel.append(i)

tfidfvectorizer = TfidfVectorizer(max_df=0.5)

traindata = tfidfvectorizer.fit_transform(train)
testdata = tfidfvectorizer.transform(test)

nb_model = MultinomialNB(alpha=0.001)
nb_model.fit(traindata, trainlabel)
predict_test = nb_model.predict(testdata)
print("多项式朴素贝叶斯文本分类的准确率为：", metrics.accuracy_score(predict_test, testlabel))
print(metrics.confusion_matrix(predict_test, testlabel, labels=labels))

ber_model = BernoulliNB(alpha=0.001)
ber_model.fit(traindata, trainlabel)
ber_predict = ber_model.predict(testdata)
print("bernoulli贝叶斯文本分类的准确率为：", metrics.accuracy_score(ber_predict, testlabel))
print(metrics.confusion_matrix(ber_predict, testlabel, labels=labels))

gauss_model = GaussianNB()
gauss_model.fit(traindata.toarray(), trainlabel)
gauss_predict = ber_model.predict(testdata.toarray())
print("GaussianNB贝叶斯文本分类的准确率为：", metrics.accuracy_score(gauss_predict, testlabel))
print(metrics.confusion_matrix(gauss_predict, testlabel, labels=labels))
