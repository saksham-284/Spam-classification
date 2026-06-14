import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from data_profiling import ProfileReport
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
df=pd.read_csv("C:/Users/SAKSHAM/Downloads/if_main_practice/python_hello/email.csv")
# print(df.shape)
df["Category"]=df["Category"].str.strip()
# print(df.isnull().sum())
# print(df["Category"].unique())
df=df[df["Category"]!='{"mode":"full"']
# print(df["Category"].value_counts())
df["Category"]=df["Category"].map({"ham":0,"spam":1})
X=df["Message"]
y=df["Category"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=Pipeline([("vectorizer",CountVectorizer()),("model",MultinomialNB())])
model.fit(X_train,y_train)
y_train_pred=model.predict(X_train)
print("train",accuracy_score(y_train,y_train_pred))
y_pred=model.predict(X_test)
print("accuracy",accuracy_score(y_test,y_pred))
print("confusion ",confusion_matrix(y_test,y_pred))
print("report",classification_report(y_test,y_pred))
