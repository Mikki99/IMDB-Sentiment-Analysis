import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Sequential
from wordcloud import WordCloud
from sklearn.metrics import roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
import itertools
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression


tsv_data = pd.read_csv('movie_reviews3.tsv', sep='\t')
tsv_data.columns = ['Ratings', 'Reviews']
tsv_data['binaryRatings'] = np.where(tsv_data['Ratings'] <= 5, -1, 1)

print("Binary Ratings")
print(tsv_data.head())

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
                 explode = (0, 0.1),
                 textprops = {"fontsize":32})
plt.show()

tsv_data = tsv_data.drop(tsv_data.loc[tsv_data['binaryRatings'] == 1].sample(frac=0.56).index)

#Visualizing Data after downsampling
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
                 explode = (0, 0.1),
                 textprops = {"fontsize":32})
plt.show()

#Data cloud for binary Rating +1 for a review
wordcloud = WordCloud(stopwords=stopwords.words('english'),
                  background_color='white',
                  colormap = 'Greens',
                  width=2500,
                  height=2000
                 ).generate(positiveRatings['Reviews'].iloc[1])
plt.figure(1,figsize=(10, 7))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#Data cloud for binary Rating -1 for a review
wordcloud = WordCloud(stopwords=stopwords.words('english'),
                  background_color='white',
                  colormap = 'Reds',
                  width=2500,
                  height=2000
                 ).generate(negativeRatings['Reviews'].iloc[2])
plt.figure(1,figsize=(10, 7))
plt.imshow(wordcloud)
plt.axis('off')
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

#########################################
# WORD CLOUD
from collections import Counter
import nltk

# Tokenizing reviews
positiveRatings["tokenized"] = positiveRatings["Reviews"].apply(nltk.word_tokenize)
negativeRatings["tokenized"] = negativeRatings["Reviews"].apply(nltk.word_tokenize)

# Creating single list with all positive words
pos_words = []
for pos_list in positiveRatings["tokenized"]:
    pos_words += pos_list

# Creating single list with all negative words
neg_words = []
for neg_list in negativeRatings["tokenized"]:
    neg_words += neg_list

# Get 100 most freq. POS words
# and store just words in a list (without count)
pos_top100 = Counter(pos_words).most_common(100)
pos_top100 = [item[0] for item in pos_top100]

# Get 100 most freq. NEG words
# and store just words in a list (without count)
neg_top100 = Counter(neg_words).most_common(100)
neg_top100 = [item[0] for item in neg_top100]

# WordCloud POS
wordcloud = WordCloud(stopwords=stopwords.words('english'),
                  background_color='white',
                  width=2500,
                  height=2000
                 ).generate(" ".join(pos_top100))
plt.figure(1,figsize=(10, 7))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# WordCloud NEG
wordcloud = WordCloud(stopwords=stopwords.words('english'),
                  background_color='white',
                  width=2500,
                  height=2000
                 ).generate(" ".join(neg_top100))
plt.figure(1,figsize=(10, 7))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#########################################
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

print("\nShowcasing Preprocessing of data for one sample review\n")

sampleReview = "This is a beautifully sculpted slow paced romantic adorable and lovable movie. Halitha shameem deserves all the appreciation. Wish to see more movies coming like this in the future"

print("Sample Review from the dataset")
print("---->",sampleReview)

sampleReviewLowercase = sampleReview.lower()
print("Lowercase conversion")
print("---->",sampleReviewLowercase)

sampleReviewPR = punctuationRemoval(sampleReviewLowercase)
print("Punctuation Removal")
print("---->",sampleReviewPR)

sampleReviewRSW = removeStopWords(sampleReviewPR)
print("Stop Words Removal")
print("---->",sampleReviewRSW)

sampleReviewStemming = stemWords(sampleReviewRSW)
print("Stemming")
print("---->",sampleReviewStemming)

sampleReviewLemmatizing = lemmatizeWords(sampleReviewStemming)
print("Lemmatizing")
print("---->",sampleReviewLemmatizing)

train, test = train_test_split(tsv_data, test_size=0.2, random_state=42, shuffle=True)
tf_idf = TfidfVectorizer()

Xtrain_tf = tf_idf.fit_transform(train['Reviews'])
Xtrain_tf = tf_idf.transform(train['Reviews'])
print("\nsamples: %d, features: %d" % Xtrain_tf.shape)

Xtest_tf = tf_idf.transform(test['Reviews'])
print("samples: %d, features: %d" % Xtest_tf.shape)

# #MultinomialNB - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
naiveBayesClassifier = MultinomialNB()
naiveBayesClassifier.fit(Xtrain_tf, train['binaryRatings'])

trainRatingPrediction = naiveBayesClassifier.predict(Xtrain_tf)
testRatingPrediction = naiveBayesClassifier.predict(Xtest_tf)

print("Training Confusion Matrix")
print(metrics.classification_report(train['binaryRatings'], trainRatingPrediction))

print("Testing Confusion Matrix")
print(metrics.classification_report(test['binaryRatings'], testRatingPrediction))

# TODO: Logistic Regression
logisticRegression_variable = LogisticRegression()
cross_score = cross_val_score(logisticRegression_variable, Xtrain_tf, train['binaryRatings'])
logisticRegression_variable.fit(Xtrain_tf, train['binaryRatings'])
test_prediction = logisticRegression_variable.predict(Xtest_tf)
print(metrics.classification_report(test['binaryRatings'], test_prediction))

YscoreLR = logisticRegression_variable.predict_proba(Xtest_tf)
fprLR, tprLR, _ = roc_curve(test['binaryRatings'], YscoreMNB[:, 1])

# Linear SVC
lin_svc = LinearSVC()
lin_svc.fit(Xtrain_tf, train['binaryRatings'])
pred_svc = lin_svc.predict(Xtest_tf)

mean_error = []
std_error = []
C = [0.0001, 0.01, 0.1, 1, 10, 50]
for c in C:
    model = LinearSVC(C=c)
    scores = cross_val_score(model, Xtrain_tf.toarray(), train['binaryRatings'], cv=5, scoring='f1')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())

plt.figure()
plt.title("Linear SVC")
plt.errorbar(C, mean_error, yerr=std_error)
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.xlim((-1, 20))
plt.show()

lin_svc = LinearSVC(C=0.1)
lin_svc.fit(Xtrain_tf, train['binaryRatings'])
pred_svc = lin_svc.predict(Xtest_tf)

print(metrics.classification_report(test['binaryRatings'], pred_svc))

#ROC
YscoreMNB = naiveBayesClassifier.predict_proba(Xtest_tf)
fprMNB, tprMNB, _ = roc_curve(test['binaryRatings'], YscoreMNB[:, 1])

YscoreLSVC = lin_svc._predict_proba_lr(Xtest_tf)
fprLSVC, tprLSVC, _ = roc_curve(test['binaryRatings'], YscoreLSVC[:, 1])

####################################################################
####################################################################
####################################################################

# Neural Networks -- LSTM
# Replace -1 with 0 labels for negative reviews
train_nn = train.replace({'binaryRatings': {-1: 0}})
test_nn = test.replace({'binaryRatings': {-1: 0}})

# Split train and test data into reviews and labels
X_train = train_nn['Reviews'].values
y_train = train_nn['binaryRatings'].values
X_test = test_nn['Reviews'].values
y_test = test_nn['binaryRatings'].values

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
print(X_train_seq_padded.shape)
print(X_test_seq_padded.shape)


def lstm_hp_tune(verbose=False):
    def build_model(hp):
        model = Sequential()

        model.add(Embedding(vocab_size,
                            hp.Int('emb_dim', min_value=32, max_value=256, step=32),
                            input_shape=(max_len, )))
        for i in range(hp.Int('n_layers', 0, 3)):
            model.add(Bidirectional(LSTM(hp.Int(f'BiLSTM_{i}_neurons', min_value=2, max_value=64, step=4),
                                         return_sequences=True)))
        model.add(Bidirectional(LSTM(units=hp.Int('BiLSTM_neurons', min_value=2, max_value=64, step=4))))
        model.add(Dense(hp.Int('dense_neurons', min_value=32, max_value=256, step=32), activation='relu'))
        model.add(Dropout(hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=3,
        executions_per_trial=1
    )

    tuner.search(x=X_train_seq_padded,
                 y=y_train,
                 epochs=5,
                 batch_size=64,
                 validation_data=(X_test_seq_padded, y_test))

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0].values

    if verbose:
        print(best_model.summary())
        print(tuner.results_summary())

    return best_hp


# best_hp = lstm_hp_tune()
# print(best_hp)

# Defining the model
inputs = Input(shape=(max_len,))
x = Embedding(10000, 64)(inputs)
x = Bidirectional(LSTM(units=22))(x)
x = Dense(96, activation="relu")(x)
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
                    batch_size=64,
                    epochs=35,
                    validation_data=(X_test_seq_padded, y_test),
                    callbacks=[EarlyStopping(monitor='val_accuracy', patience=3)])

model.save("LSTM_model_tuned")
model = tf.keras.models.load_model("LSTM_model_tuned")
# Predictions
preds_test = model.predict(X_test_seq_padded)
preds_test = np.array([1 if pred > 0.5 else 0 for pred in preds_test.flatten()])

preds_train = model.predict(X_train_seq_padded)
preds_train = np.array([1 if pred > 0.5 else 0 for pred in preds_train.flatten()])

print(metrics.classification_report(y_test, preds_test))
print(metrics.classification_report(y_train, preds_train))

result_list = []
outcome_labels = []
for index, row in tsv_data.iterrows():
    result_textBlob = TextBlob(row["Reviews"]).sentiment
    polarity_value = result_textBlob.polarity
    if polarity_value > 0:
        outcome_labels.append(-1)
    else:
        outcome_labels.append(1)
print(test['binaryRatings'])
# print("Inbult Function",f1_score(test['binaryRatings'], outcome_labels))
# print("Our Model",f1_score(tsv_data['binaryRatings'],test['binaryRatings']))

# Random baseline classifier
dummy = DummyClassifier(strategy='most_frequent').fit(Xtrain_tf, train["binaryRatings"])
dummy_preds = dummy.predict(Xtest_tf)
fprDUMMY, tprDUMMY, _ = roc_curve(test["binaryRatings"],
                            dummy.predict_proba(Xtest_tf)[:, 1])
# ROC curves
fprLSTM, tprLSTM, _ = roc_curve(test['binaryRatings'], model.predict(X_test_seq_padded).ravel())

plt.plot(fprMNB, tprMNB)
plt.plot(fprLSVC, tprLSVC)
plt.plot(fprLSTM, tprLSTM)
plt.plot(fprDUMMY, tprDUMMY, linestyle='--')
plt.plot(fprLR, tprLR)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC Plot')
plt.legend(['Naive Bayes', 'Linear SVC', 'LSTM', 'Baseline CLF', 'Logistic Regression'])
plt.show()

# Confusion matrices
conf_mtx_lstm = confusion_matrix(test_nn["binaryRatings"], preds_test)
conf_mtx_linSVC = confusion_matrix(test["binaryRatings"], pred_svc)
conf_mtx_NB = confusion_matrix(test["binaryRatings"], testRatingPrediction)
conf_mtx_dummy = confusion_matrix(test["binaryRatings"], dummy_preds)
conf_mtx_lr = confusion_matrix(test["binaryRatings"], test_prediction)


def plot_conf_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

labels = ['positive', 'negative']

plot_conf_matrix(cm=conf_mtx_lstm, classes=labels, title="LSTM")
plot_conf_matrix(cm=conf_mtx_linSVC, classes=labels, title="Linear SVC")
plot_conf_matrix(cm=conf_mtx_NB, classes=labels, title="Naive Bayes")
plot_conf_matrix(cm=conf_mtx_dummy, classes=labels, title="Baseline CLF")
plot_conf_matrix(cm=conf_mtx_lr, classes=labels, title="Logistic Regression")

plt.show()

