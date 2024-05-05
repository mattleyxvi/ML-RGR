import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np

data = pd.read_csv('data\diamonds.csv')
data = data.drop(columns = ['Unnamed: 0'])
correlation_matrix = data.corr(numeric_only = True)

st.header("Соотношение количества алмазов различного цвета")
fig=plt.figure()
size=data.groupby("color").size()
plt.pie(size.values,labels=size.index,autopct='%1.0f%%')
st.pyplot(plt)

st.header("Корреляция стоимости c числовыми столбцами")
fig=plt.figure()
fig.add_subplot(sns.heatmap(correlation_matrix,annot=True,cmap="YlGnBu", fmt=".2f"))
st.pyplot(fig)

st.header("Boxplot о распределении размеров по трем осям")
fig=plt.figure()
plt.boxplot(data[['x','y','z']],labels=['x','y','z'])
st.pyplot(plt)

st.header("Histogram, показывающая количество различных значений прозрачности ")
fig=plt.figure()
plt.hist(data['clarity'])
st.pyplot(plt)