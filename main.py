import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis




@st.cache
def split_data(df):
    features= df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    labels = df['variety'].values

    return  train_test_split(features, labels, train_size=0.7, random_state=1)

def getData():
    SL = st.slider("Sepal length", 2,25, 3)
    SW = st.slider("Sepal width", 2,25, 3)
    PL = st.slider("Petal length", 2,25, 3)
    PW = st.slider("Petal width", 2,25, 3)
    print(f"LOG: the prediction input is: {[SL, SW, PL, PW]}")
    return np.array([SL, SW, PL, PW]).reshape(1,-1)

def main(df):
    X_train,X_test, y_train, y_test = split_data(df)
    alg = ["Decision Tree", "Support Vector Machine", "KNeighborsClassifier", "Linear SVC", "SVC", "GaussianPro00cessClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "MLPClassifier", "AdaBoostClassifier", "GaussianNB", "QuadraticDiscriminantAnalysis"]

    classifier = st.selectbox('Which algorithm?', alg)

    if classifier=='Decision Tree':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        cm_dtc=confusion_matrix(y_test,pred_dtc)
        st.write('Confusion matrix: ', cm_dtc)
        input = getData()
        st.write('The classification is: ', dtc.predict(input)[0])

    elif classifier == 'Support Vector Machine':
        svm=SVC()
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_svm = svm.predict(X_test)
        cm=confusion_matrix(y_test,pred_svm)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', svm.predict(input)[0])

    elif classifier == "KNeighborsClassifier":
        clf = KNeighborsClassifier(3)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "Linear SVC":
        clf = SVC(kernel="linear", C=0.025)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "SVC":
        clf = SVC(gamma=2, C=1)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "GaussianProcessClassifier":
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "RandomForestClassifier":
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "MLPClassifier":
        clf = MLPClassifier(alpha=1, max_iter=1000)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "AdaBoostClassifier":
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "GaussianNB":
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])

    elif classifier == "QuadraticDiscriminantAnalysis":
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_clf = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred_clf)
        st.write('Confusion matrix: ', cm)
        input = getData()
        st.write('The classification is: ', clf.predict(input)[0])
    
@st.cache
def loadData():
    return pd.read_csv("iris.csv")
    

df = loadData()
st.title('Iris')

if st.checkbox('Show dataframe'):
    st.write(df)

st.subheader('Scatter plot')
species = st.multiselect('Show iris per variety?', df['variety'].unique())
col1 = st.selectbox('Which feature on x?', df.columns[0:4])
col2 = st.selectbox('Which feature on y?', df.columns[0:4])
new_df = df[(df['variety'].isin(species))]
st.write(new_df)

# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2, color='variety')

# Plot!
st.plotly_chart(fig)
st.subheader('Histogram')
feature = st.selectbox('Which feature?', df.columns[0:4])

# Filter dataframe
new_df2 = df[(df['variety'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color="variety", marginal="rug")
st.plotly_chart(fig2)
st.subheader('Machine Learning models')

main(df)