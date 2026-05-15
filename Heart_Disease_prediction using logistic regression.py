import streamlit as st
import pickle
import streamlit.components.v1 as comp

st.title("Heart Disease Prediction")

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.datasets import make_classification





heart = pd.read_csv("F:/Sem 5/Notes/ml/ML_Project/heart_dataset/framingham.csv")
st.write("Represnting Original Data")
heart
st.write(" ")
st.write("Information about Data Columns")
st.write(heart.info())

st.write("Descriptive of Data")

st.write(heart.describe())

st.write("Data Cleaning")
st.write(heart.dropna(inplace = True) )


x = heart.drop('TenYearCHD', axis=1)
y = heart['TenYearCHD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

st.subheader("Scores")

st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
st.write(f"\nConfusion Matrix: \n ", confusion_matrix(y_test, y_pred))
st.write(f"\nPrecision : {precision_score(y_test, y_pred)}")
st.write(f"\nRecall :   {recall_score(y_test, y_pred)}")
st.write(f"\nF1-Score : {f1_score(y_test, y_pred)}")
st.write(f"\nROC-AUC :  {roc_auc_score(y_test, y_pred)}")


st.subheader("Graphs for Logistic Regression")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot= True, fmt= "d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color= "darkorange", lw=2, label=f"ROC curse (AUC = { roc_auc: .2f})")
ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle= "--")
ax.set_xlim([0.0, 1.0])
ax.set_xlim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receive Operating Characteristic")
ax.legend(loc="lower right")
st.pyplot(fig)



