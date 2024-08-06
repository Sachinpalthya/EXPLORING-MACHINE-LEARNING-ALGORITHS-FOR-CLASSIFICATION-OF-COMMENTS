from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
#nltk.download()
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss

main = Tk()
main.title("Exploring Machine Learning Algorithms for classification of online toxic comments")
main.geometry("1300x1200")

global filename
global X, Y1, Y2, Y3, Y4, Y5, Y6
accuracy = []
global X_train, X_test, y_train, y_test
loss = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
global classifier1,classifier2,classifier3,classifier4,classifier5,classifier6
global tfidf_vectorizer

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n")
    

def preprocess():
    textdata.clear()
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    Y5 = []
    Y6 = []
    text.delete('1.0', END)
    dataset = pd.read_csv(filename,encoding='iso-8859-1',nrows = 300)
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'comment_text')
        toxic = dataset.get_value(i, 'toxic')
        severe_toxic = dataset.get_value(i, 'severe_toxic')
        obscene = dataset.get_value(i, 'obscene')
        threat = dataset.get_value(i, 'threat')
        insult = dataset.get_value(i, 'insult')
        identity_hate = dataset.get_value(i, 'identity_hate')
        msg = str(msg)
        msg = msg.strip().lower()
        Y1.append(int(toxic))
        Y2.append(int(severe_toxic))
        Y3.append(int(obscene))
        Y4.append(int(threat))
        Y5.append(int(insult))
        Y6.append(int(identity_hate))
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+"\n")
    Y1 = np.asarray(Y1)        
    Y2 = np.asarray(Y2)
    Y3 = np.asarray(Y3)
    Y4 = np.asarray(Y4)
    Y5 = np.asarray(Y5)
    Y6 = np.asarray(Y6)

def countVector():
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    global tfidf_vectorizer
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y1 = Y1[indices]
    Y2 = Y2[indices]
    Y3 = Y3[indices]
    Y4 = Y4[indices]
    Y5 = Y5[indices]
    Y6 = Y6[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2,random_state=0)
    text.insert(END,"\n\nTotal Comments found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(X_test))+"\n")
    

def train(Xdata,Ydata,cls):
    X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.2,random_state=0)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    acc = accuracy_score(y_test,predict)*100
    loss = hamming_loss(y_test,predict)*100
    return acc,loss


def train1(Xdata,Ydata,cls):
    X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.2,random_state=0)
    cls.fit(Xdata, Ydata)
    predict = cls.predict(X_test) 
    acc = accuracy_score(y_test,predict)*100
    loss = hamming_loss(y_test,predict)*100
    return acc,loss
    

def runSVM():
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    text.delete('1.0', END)
    accuracy.clear()
    loss.clear()
    cls1 = svm.SVC()
    acc1,loss1 = train(X,Y1,cls1)
    cls2 = svm.SVC()
    acc2,loss2 = train(X,Y2,cls2)
    cls3 = svm.SVC()
    acc3,loss3 = train(X,Y3,cls3)
    cls4 = svm.SVC()
    acc4,loss4 = train(X,Y4,cls4)
    cls5 = svm.SVC()
    acc5,loss5 = train(X,Y5,cls5)
    cls6 = svm.SVC()
    acc6,loss6 = train(X,Y6,cls6)
    acc = acc1 + acc2 + acc3 + acc4 + acc5 + acc6
    acc = (acc / 600) * 100
    lossValue = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    accuracy.append(acc)
    lossValue = lossValue / 600
    loss.append(lossValue)
    text.insert(END,"SVM Accuracy : "+str(acc)+"\n")
    text.insert(END,"SVM Hamming Loss : "+str(lossValue)+"\n\n")
    

def runLR():
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    cls1 = LogisticRegression()
    acc1,loss1 = train(X,Y1,cls1)
    cls2 = LogisticRegression()
    acc2,loss2 = train(X,Y2,cls2)
    cls3 = LogisticRegression()
    acc3,loss3 = train(X,Y3,cls3)
    cls4 = LogisticRegression()
    acc4,loss4 = train(X,Y4,cls4)
    cls5 = LogisticRegression()
    acc5,loss5 = train(X,Y5,cls5)
    cls6 = LogisticRegression()
    acc6,loss6 = train(X,Y6,cls6)
    acc = acc1 + acc2 + acc3 + acc4 + acc5 + acc6
    acc = (acc / 600) * 100
    accuracy.append(acc)
    lossValue = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    lossValue = lossValue / 600
    loss.append(lossValue)
    text.insert(END,"Logistic Regression Accuracy : "+str(acc)+"\n")
    text.insert(END,"Logistic Regression Hamming Loss : "+str(lossValue)+"\n\n")
    
    
def runNB():
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    cls1 = GaussianNB()
    acc1,loss1 = train(X,Y1,cls1)
    cls2 = GaussianNB()
    acc2,loss2 = train(X,Y2,cls2)
    cls3 = GaussianNB()
    acc3,loss3 = train(X,Y3,cls3)
    cls4 = GaussianNB()
    acc4,loss4 = train(X,Y4,cls4)
    cls5 = GaussianNB()
    acc5,loss5 = train(X,Y5,cls5)
    cls6 = GaussianNB()
    acc6,loss6 = train(X,Y6,cls6)
    acc = acc1 + acc2 + acc3 + acc4 + acc5 + acc6
    acc = (acc / 600) * 100
    accuracy.append(acc)
    lossValue = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    lossValue = lossValue / 600
    loss.append(lossValue)
    text.insert(END,"Naive Bayes Accuracy : "+str(acc)+"\n")
    text.insert(END,"Naive Bayes Hamming Loss : "+str(lossValue)+"\n\n")

def runDecisionTree():
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    cls1 = DecisionTreeClassifier()
    acc1,loss1 = train(X,Y1,cls1)
    cls2 = DecisionTreeClassifier()
    acc2,loss2 = train(X,Y2,cls2)
    cls3 = DecisionTreeClassifier()
    acc3,loss3 = train(X,Y3,cls3)
    cls4 = DecisionTreeClassifier()
    acc4,loss4 = train(X,Y4,cls4)
    cls5 = DecisionTreeClassifier()
    acc5,loss5 = train(X,Y5,cls5)
    cls6 = DecisionTreeClassifier()
    acc6,loss6 = train(X,Y6,cls6)
    acc = acc1 + acc2 + acc3 + acc4 + acc5 + acc6
    acc = (acc / 600) * 100
    accuracy.append(acc)
    lossValue = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    lossValue = lossValue / 600
    loss.append(lossValue)
    text.insert(END,"Decision Tree Accuracy : "+str(acc)+"\n")
    text.insert(END,"Decision Tree Hamming Loss : "+str(lossValue)+"\n\n")
    
def runRandomForest():
    global classifier1,classifier2,classifier3,classifier4,classifier5,classifier6
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    cls1 = RandomForestClassifier()
    acc1,loss1 = train1(X,Y1,cls1)
    cls2 = RandomForestClassifier()
    acc2,loss2 = train1(X,Y2,cls2)
    cls3 = RandomForestClassifier()
    acc3,loss3 = train1(X,Y3,cls3)
    cls4 = RandomForestClassifier()
    acc4,loss4 = train1(X,Y4,cls4)
    cls5 = RandomForestClassifier()
    acc5,loss5 = train1(X,Y5,cls5)
    cls6 = RandomForestClassifier()
    acc6,loss6 = train1(X,Y6,cls6)
    classifier1 = cls1
    classifier2 = cls2
    classifier3 = cls3
    classifier4 = cls4
    classifier5 = cls5
    classifier6 = cls6
    acc = acc1 + acc2 + acc3 + acc4 + acc5 + acc6
    acc = (acc / 600) * 100
    accuracy.append(acc)
    lossValue = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    lossValue = lossValue / 600
    loss.append(lossValue)
    text.insert(END,"Random Forest Accuracy : "+str(acc)+"\n")
    text.insert(END,"Random Forest Hamming Loss : "+str(lossValue)+"\n\n")

def runKNN():
    global X, Y1, Y2, Y3, Y4, Y5, Y6
    cls1 = KNeighborsClassifier(n_neighbors = 2)
    acc1,loss1 = train(X,Y1,cls1)
    cls2 = KNeighborsClassifier(n_neighbors = 2)
    acc2,loss2 = train(X,Y2,cls2)
    cls3 = KNeighborsClassifier(n_neighbors = 2)
    acc3,loss3 = train(X,Y3,cls3)
    cls4 = KNeighborsClassifier(n_neighbors = 2)
    acc4,loss4 = train(X,Y4,cls4)
    cls5 = KNeighborsClassifier(n_neighbors = 2)
    acc5,loss5 = train(X,Y5,cls5)
    cls6 = KNeighborsClassifier(n_neighbors = 2)
    acc6,loss6 = train(X,Y6,cls6)
    acc = acc1 + acc2 + acc3 + acc4 + acc5 + acc6
    acc = (acc / 600) * 100
    accuracy.append(acc)
    lossValue = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    lossValue = lossValue / 600
    loss.append(lossValue)
    text.insert(END,"KNN Accuracy : "+str(acc)+"\n")
    text.insert(END,"KNN Hamming Loss : "+str(lossValue)+"\n\n")
    

def graph():
    df = pd.DataFrame([['SVM','Accuracy',accuracy[0]],['SVM','Hamming Loss',loss[0]],
                       ['Logistic Regression','Accuracy',accuracy[1]],['Logistic Regression','Hamming Loss',loss[1]],
                       ['Naive Bayes','Accuracy',accuracy[2]],['Naive Bayes','Hamming Loss',loss[2]],
                       ['Decision Tree','Accuracy',accuracy[3]],['Decision Tree','Hamming Loss',loss[3]],
                       ['Random Forest','Accuracy',accuracy[4]],['Random Forest','Hamming Loss',loss[4]],
                       ['KNN','Accuracy',accuracy[5]],['KNN','Hamming Loss',loss[5]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()


def predict():
    global classifier1,classifier2,classifier3,classifier4,classifier5,classifier6
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(testfile)
    text.delete('1.0', END)
    for i in range(len(testData)):
        msg = testData.get_value(i, 'comment')
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = tfidf_vectorizer.transform([review]).toarray()
        predict1 = classifier1.predict(testReview)
        predict2 = classifier2.predict(testReview)
        predict3 = classifier3.predict(testReview)
        predict4 = classifier4.predict(testReview)
        predict5 = classifier5.predict(testReview)
        predict6 = classifier6.predict(testReview)
        if predict1 == 0 and predict2 == 0 and predict3 == 0 and predict4 == 0 and predict5 == 0 and predict6 == 0:
            text.insert(END,msg+" [NOT CONTAINS TOXIC Comments]\n\n")
        elif predict1 == 1:
            text.insert(END,msg+" [CONTAINS TOXIC Comments]\n\n")
        elif predict2 == 1:
            text.insert(END,msg+" [CONTAINS SEVERE_TOXIC Comments]\n\n")
        elif predict3 == 1:
            text.insert(END,msg+" [CONTAINS OBSCENE Comments]\n\n")
        elif predict4 == 1:
            text.insert(END,msg+" [CONTAINS THREAT Comments]\n\n")
        elif predict5 == 1:
            text.insert(END,msg+" [CONTAINS INSULT Comments]\n\n")
        elif predict6 == 1:
            text.insert(END,msg+" [CONTAINS IDENTITY_HATE Comments]\n\n")    
            
    
font = ('times', 15, 'bold')
title = Label(main, text='Exploring Machine Learning Algorithms for classification of online toxic comments')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Toxic Comments Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

cvButton = Button(main, text="Apply Count Vectorizer", command=countVector)
cvButton.place(x=20,y=200)
cvButton.config(font=ff)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=20,y=250)
svmButton.config(font=ff)

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLR)
lrButton.place(x=20,y=300)
lrButton.config(font=ff)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB)
nbButton.place(x=20,y=350)
nbButton.config(font=ff)

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=20,y=400)
dtButton.config(font=ff)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=20,y=450)
rfButton.config(font=ff)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=20,y=500)
knnButton.config(font=ff)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=20,y=550)
graphButton.config(font=ff)

predictButton = Button(main, text="Predict Toxic Comments from Test Data", command=predict)
predictButton.place(x=20,y=600)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
