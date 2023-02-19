import os
from preprocessMethod import preprocessData
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

filenames=os.listdir(r'D:/360MoveData/Users/31365/Desktop/dataset')
count = 1
index = []
train = []
lable = []
for i in filenames:
    index.append(i)
    path = 'D:/360MoveData/Users/31365/Desktop/dataset/'+i
    count += 1
    with open(path, "r", encoding='utf-8') as file:
        train.append(preprocessData(file))
        lable.append(i)



countvectorizer = CountVectorizer()
tfidfvectorizer = TfidfVectorizer()
count_wm = countvectorizer.fit_transform(train)
tfidf_wm = tfidfvectorizer.fit_transform(train)
count_tokens = countvectorizer.get_feature_names_out()
tfidf_tokens = tfidfvectorizer.get_feature_names_out()
df_countvect = pd.DataFrame(data = count_wm.toarray(),index = index,columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = index,columns = tfidf_tokens)
print("Count Vectorizer\n")
print(df_countvect)
print("\nTD-IDF Vectorizer\n")
print(df_tfidfvect)

nb_model = MultinomialNB(alpha=0.001)
nb_model.fit(tfidf_wm,lable)







