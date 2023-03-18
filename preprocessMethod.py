from nltk.corpus import *
from nltk.stem import *
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import re
def preprocessData(text):
    text = text.lower().split()
    new_text = ""
    for word in text:
        if word not in stopwords.words('english'):
            new_text = new_text + " " + word

    symbols = "!\"#$%&()*+-.,/:;<=>?@[\]^_`'{|}~\n"
    for i in symbols:
        new_text = np.char.replace(new_text, i, '')

    new_text = str(new_text).split()

    stemmer = PorterStemmer()
    st = LancasterStemmer()
    new_text = [stemmer.stem(word) for word in new_text]


    regex1 = re.compile(r'\d')
    regex2 = re.compile(r'\w*\d')
    filtered1 = [i for i in new_text if not regex1.match(i)]
    filtered2 = [i for i in filtered1 if not regex2.match(i)]
    result = [i for i in filtered2 if not len(i) == 1]

    str1 = ""
    for ele in result:
        str1 += ele + ' '

    return str1
