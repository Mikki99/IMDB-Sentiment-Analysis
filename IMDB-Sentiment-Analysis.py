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
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

tsv_data = pd.read_csv('movie_reviews3.tsv', sep='\t')
tsv_data.columns = ['Ratings','Reviews']
tsv_data['binaryRatings'] = np.where(tsv_data['Ratings'] <= 5, -1, 1)

print("Binary Ratings")
print(tsv_data.head())

tsv_data.to_csv('labelledData3.tsv', encoding='utf-8', index=False)

tsv_data = tsv_data.drop(tsv_data.loc[tsv_data['binaryRatings'] == 1].sample(frac=0.56).index)

#Visualizing Data
figure = plt.figure(figsize=(5,5))
positiveRatings = tsv_data[tsv_data['binaryRatings'] == 1]
negativeRatings = tsv_data[tsv_data['binaryRatings'] == -1]
count = [positiveRatings['binaryRatings'].count(), negativeRatings['binaryRatings'].count()]
colors = ["lightgreen",'red']
pieChart = plt.pie(count,
                 labels=["Positive Reviews", "Negative Reviews"],
                 autopct ='%1.1f%%',
                 colors = colors,
                 shadow = True,
                 startangle = 45,
                 explode = (0, 0.1))
plt.show()

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

tsv_data.to_csv('PreprocessedData3.tsv', encoding='utf-8', index=False)

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

trainRatingPrediction = naiveBayesClassifier.predict(Xtrain_tf)
testRatingPrediction = naiveBayesClassifier.predict(Xtest_tf)

print("Training Confusion Matrix")
print(metrics.classification_report(train['binaryRatings'], trainRatingPrediction))

print("Testing Confusion Matrix")
print(metrics.classification_report(test['binaryRatings'], testRatingPrediction))

# TODO: Logistic Regression


# Linear SVC
lin_svc = LinearSVC()
lin_svc.fit(Xtrain_tf, train['binaryRatings'])
pred_svc = lin_svc.predict(Xtest_tf)

print(metrics.classification_report(test['binaryRatings'], pred_svc))

# Just playing around with another model
# XG-Boost
clf3 = XGBClassifier()
clf3.fit(Xtrain_tf, train['binaryRatings'])
preds3 = clf3.predict(Xtest_tf)

print(metrics.classification_report(test['binaryRatings'], preds3))

####################################################################
####################################################################
####################################################################

# Neural Networks -- LSTM
# Replace -1 with 0 labels for negative reviews
train_nn = train.replace({'binaryRatings': {-1: 0}})
test_nn = test.replace({'binaryRatings': {-1: 0}})


# Split train and test data into reviews and labels
X_train = train['Reviews'].values
y_train = train['binaryRatings'].values
X_test = test['Reviews'].values
y_test = test['binaryRatings'].values

# Fit Keras tokenizer on train reviews
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

# Encode train and test reviews
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Some statistics about the reviews
l = np.array([len(i) for i in X_train_seq])
print('minimum number of words in a review: {}'.format(l.min()))
print('median number of words in a review: {}'.format(np.median(l)))
print('average number of words in a review: {}'.format(l.mean()))
print('maximum number of words in a review: {}'.format(l.max()))

# Limit review length to max_len tokens
max_len = 385

# Use padding so that all sequences are of same length
X_train_seq_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=max_len)


# Defining the model
inputs = Input(shape=(max_len,))
x = Embedding(vocab_size, 338)(inputs)
x = LSTM(units=8)(x)
x = Dense(111, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = Adam(learning_rate=0.0009)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(x=X_train_seq_padded,
                    y=y_train,
                    batch_size=56,
                    epochs=35,
                    validation_data=(X_test_seq_padded, y_test),
                    callbacks=[EarlyStopping(monitor='val_accuracy', patience=3)])

# Predictions
preds_test = model.predict(X_test_seq_padded)
preds_test = np.array([1 if pred > 0.5 else 0 for pred in preds_test.flatten()])

preds_train = model.predict(X_train_seq_padded)
preds_train = np.array([1 if pred > 0.5 else 0 for pred in preds_train.flatten()])

print(metrics.classification_report(y_test, preds_test))
print(metrics.classification_report(y_train, preds_train))
