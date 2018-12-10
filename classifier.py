
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from nltk.metrics import accuracy
import os
from TideneReadCorpus import *

PATH = "/home/bruno/base-wipo/base-total/preprocess_token/"

PATH = "/home/bruno/base-wipo/base-total-300/preprocess_stop/"
teste = "teste.csv"
treinamento = "treinamento.csv"


def main():
    #X_test = TideneIterCSVClass(PATH+teste)
    #X_train = TideneIterCSVClass(PATH+treinamento)

    X_train = TideneIterCSVTaggingExtraction(PATH+treinamento)
    X_test = TideneIterCSVTaggingExtraction(PATH+teste)

    
    Y_test = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+teste),
                        header=0,delimiter=";",usecols=["section"], quoting=3)
    
    Y_train = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+treinamento),
                        header=0,delimiter=";",usecols=["section"], quoting=3)

    #Estatistica
    sections = ["A","B","C","D","E","F","G","H"]
    print("Conjunto de treinamento ...")
    result = [(x, Y_train['section'].tolist().count(x)) for x in sections]
    print(result)

    print("Conjunto de teste ...")
    result = [(x, Y_test['section'].tolist().count(x)) for x in sections]
    print(result)

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import confusion_matrix

    tfidf_transformer = TfidfVectorizer()

    #------------ SVC test ---------------------
    X_train = TideneIterCSVTaggingExtraction(PATH+treinamento)
    X_test = TideneIterCSVTaggingExtraction(PATH+teste)
    clf = LinearSVC(C=0.177828).fit(tfidf_transformer.fit_transform(X_train), Y_train['section'].tolist())
    
    
    predict = clf.predict(tfidf_transformer.transform(X_test))

    print(accuracy(Y_test['section'].tolist(),predict))

    cm = confusion_matrix(Y_test['section'].tolist(), predict, labels = sections)
    print(cm)
    
if __name__ == "__main__":
    main()