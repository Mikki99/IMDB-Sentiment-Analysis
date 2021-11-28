import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

tsv_data = pd.read_csv('movie_reviews.tsv', sep='\t')
tsv_data.columns = ['Ratings','Reviews']
tsv_data['binaryRatings'] = np.where(tsv_data['Ratings'] <=5, -1, 1)

print("Binary Ratings")
print(tsv_data.head())

tsv_data.to_csv('labelledData.tsv', encoding='utf-8', index=False)
tsv_data['Reviews'] = tsv_data['Reviews'].str.lower()

print("Convert data to lowercase")
print(tsv_data.head())

punctuations = string.punctuation
def punctuationRemoval(text):
    return text.translate(str.maketrans('', '', punctuations))

tsv_data['Reviews'] = tsv_data['Reviews'].str.replace('\d+', '')

print("Remove numerical data")
print(tsv_data.head())

tsv_data["Reviews"] = tsv_data["Reviews"].apply(lambda text: punctuationRemoval(text))

print("Remove punctuations")
print(tsv_data.head())

stopWords = set(stopwords.words('english'))
def removeStopWords(text):
    return " ".join([word for word in str(text).split() if word not in stopWords])

tsv_data['Reviews'] = tsv_data['Reviews'].apply(lambda text: removeStopWords(text))

print("Remove stop words")
print(tsv_data.head())

stemmer = PorterStemmer()
def stemWords(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

tsv_data["Reviews"] = tsv_data["Reviews"].apply(lambda text: stemWords(text))

print("Stemming")
print(tsv_data.head())

lemmatizer = WordNetLemmatizer()
wordnetMap = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatizeWords(text):
    taggedText = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnetMap.get(position[0], wordnet.NOUN)) for word, position in taggedText])

tsv_data["Reviews"] = tsv_data["Reviews"].apply(lambda text: lemmatizeWords(text))

print("Lemmatize")
print(tsv_data.head())

tsv_data.to_csv('PreprocessedData.tsv', encoding='utf-8', index=False)

train, test = train_test_split(tsv_data, test_size=0.2, random_state=42, shuffle=True)
tf_idf = TfidfVectorizer()

Xtrain_tf = tf_idf.fit_transform(train['Reviews'])
Xtrain_tf = tf_idf.transform(train['Reviews'])
print("samples: %d, features: %d" % Xtrain_tf.shape)

Xtest_tf = tf_idf.transform(test['Reviews'])
print("samples: %d, features: %d" % Xtest_tf.shape)

#MultinomialNB - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
naiveBayesClassifier = MultinomialNB()
naiveBayesClassifier.fit(Xtrain_tf, train['binaryRatings'])
ratingPrediction = naiveBayesClassifier.predict(Xtest_tf)

print(metrics.classification_report(test['binaryRatings'], ratingPrediction))
