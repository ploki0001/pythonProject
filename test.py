import os
import linecache
from preprocessMethod import preprocessData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn import metrics
from matplotlib import pyplot as plt

filenames = os.listdir(r'D:/360MoveData/Users/31365/Desktop/dataset')
texts = []
labels=[]
for i in filenames:
    path = 'D:/360MoveData/Users/31365/Desktop/dataset/' + i
    with open(path, "r", encoding='utf-8') as file:
        count = 1
        lines = len(file.readlines())
        for count in range(lines+1):
                a = preprocessData(linecache.getline(path, count).strip())
                texts.append(a)
                labels.append(i)



######，我们可以新闻样本转化为神经网络训练所用的张量。
# 所用到的Keras库是keras.preprocessing.text.Tokenizer和keras.preprocessing.sequence.pad_sequences。代码如下所示
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences)
# from keras.utils import np_utils
# labels = np_utils.to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)



# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_new = []
for i in indices:
    labels_new.append(labels[i])

nb_validation_samples = int(0.8 * data.shape[0])

x_train = data[:-nb_validation_samples]
x_train = np.array(x_train)

y_train = labels_new[:-nb_validation_samples]
y_train = np.array(y_train)

x_val = data[-nb_validation_samples:]
x_val = np.array(x_val)

y_val = labels_new[-nb_validation_samples:]
y_val = np.array(y_val)

print(x_train[0])

###############读取词向量

embeddings_index = {}
f = open(os.path.join('D:/360MoveData/Users/31365/Desktop', 'glove.6B.100d.txt'),'r',encoding='utf-8')
for line in f.readlines():
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#############我们可以根据得到的字典生成上文所定义的词向量矩阵
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix)
#########我们将这个词向量矩阵加载到Embedding层中，注意，我们设置trainable=False使得这个编码层不可再训练。
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=10036,

                            trainable=False)


from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
print(x_train.shape)
sequence_input = Input(shape=(None, 30), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# happy learning!
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=5, batch_size=20)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
model.evaluate(x_val, y_val)
