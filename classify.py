import re

import numpy
from nltk import word_tokenize
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

cachedStopWords = stopwords.words("english")


def collection_stats():
    print("Corpus stats: ")
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");

    train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
    print(str(len(train_docs)) + " total train documents");

    test_docs = list(filter(lambda doc: doc.startswith("test"), documents));
    print(str(len(test_docs)) + " total test documents");

    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens));
    return filtered_tokens


# Return the representer, without transforming
def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90, max_features=1000, use_idf=True,
                            sublinear_tf=True);
    tfidf.fit(docs);
    return tfidf;


def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    feature_dict = {}
    for index in doc_representation.nonzero()[1]:
        feature_dict[features[index]] = doc_representation[0, index]

    # return [(features[index]: doc_representation[0, index]) for index in doc_representation.nonzero()[1]]
    return feature_dict


def log_results(classes, precision, recall, f1score):
    f = open("results.csv", "w")
    line = ''
    for cl in classes:
        line += str(cl) + ','
    f.write(line + '\n')

    line = ''
    for pr in precision:
        line += str(pr) + ','
    f.write(line + '\n')

    line = ''
    for rc in recall:
        line += str(rc) + ','
    f.write(line + '\n')

    line = ''
    for f1sc in f1score:
        line += str(f1sc) + ','
    f.write(line + '\n')

    f.close()


def main():
    collection_stats()

    print("Staring classifier ..")

    X_train = list()
    X_test = list()

    y_train = list()
    y_test = list()

    print("Reading training and testing data ..")

    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            X_train.append(reuters.raw(doc_id))
            y_train.append(reuters.categories(doc_id))
        else:
            X_test.append(reuters.raw(doc_id))
            y_test.append(reuters.categories(doc_id))

    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)
    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)

    binarizer = MultiLabelBinarizer(classes=reuters.categories())

    classifier = Pipeline([
        ('vectorizer',
         TfidfVectorizer(tokenizer=tokenize, min_df=0, max_df=0.90, max_features=3000, use_idf=True,
                         sublinear_tf=True)),
        # ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))])
    print("Training classifier ..")
    classifier.fit(X_train, binarizer.fit_transform(y_train))
    print("Testing classifier ..")
    res = classifier.predict(X_test)

    hard_precision = classifier.score(X_test, binarizer.transform(y_test))

    precision = average_precision_score(res, binarizer.fit_transform(y_test), average=None)
    recall = recall_score(res, binarizer.fit_transform(y_test), average=None)
    f1score = f1_score(res, binarizer.fit_transform(y_test), average=None)
    print("Hard precision: " + str(hard_precision))

    log_results(reuters.categories(), precision, recall, f1score)


if __name__ == '__main__':
    main()
