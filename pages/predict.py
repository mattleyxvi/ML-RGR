import streamlit as st
import pandas as pd
import pickle
import keras

data1 = pd.read_csv('data/dmds.csv')
data = pd.read_csv('data/dmd_prcd.csv')
st.header('Введенные вами данные:')


with open("models/Linreg_model.pkl","rb") as f:
    linReg=pickle.load(f)
with open("models/Bagging_model.pkl","rb") as f:
    baggingReg=pickle.load(f)
with open("models/GBoosting_model.pkl","rb") as f:
    gradientBoostingReg=pickle.load(f)
with open("models/Stacking_model.pkl","rb") as f:
    stackingReg=pickle.load(f)
neoroReg=keras.models.load_model('models/reg_model.keras')

color = st.sidebar.selectbox('color',('J','I', 'H', 'G','F','E', 'D'))
clarity = st.sidebar.selectbox('clarity',('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))
cut = st.sidebar.selectbox('cut',('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'))

x = st.sidebar.slider('x',data['x'].sort_values()[0],data['x'].sort_values().to_list()[-1])
y = st.sidebar.slider('y',data['y'].sort_values()[0],data['y'].sort_values().to_list()[-1])
z = st.sidebar.slider('z',data['z'].sort_values()[0],data['z'].sort_values().to_list()[-1])
table = st.sidebar.slider('table',data['table'].sort_values()[0],data['table'].sort_values().to_list()[-1])
depth = st.sidebar.slider('depth',data['depth'].sort_values()[0],data['depth'].sort_values().to_list()[-1])
carat = st.sidebar.slider('carat',data['carat'].sort_values()[0],data['carat'].sort_values().to_list()[-1])


val_dict = {
    'J':0,'I':1, 'H':2, 'G':3,'F':4,'E':5, 'D':6,
    'I1':0, 'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7,
    'Fair':0, 'Good':1, 'Very Good':2, 'Premium':3, 'Ideal':4
}
data_dict = {
    'carat':carat,
    'cut':val_dict[cut],
    'color':val_dict[color],
    'clarity':val_dict[clarity],
    'depth':depth,
    'table':table,
    'x':x,
    'y':y,
    'z':z,
}
data_val_dict = {
    'carat':carat,
    'cut':data['cut'].sort_values()[val_dict[cut]],
    'color':data['color'].sort_values()[val_dict[color]],
    'clarity':data['clarity'].sort_values()[val_dict[clarity]],
    'depth':depth,
    'table':table,
    'x':x,
    'y':y,
    'z':z,
}
col = data.columns.to_list()
col.remove('price')
value = pd.DataFrame([data_val_dict])
st.write(pd.DataFrame([data_dict]))


getPredButton=st.button("Получить предсказание")
if getPredButton:
    st.header("Результат прогноза:")
    linReg_result=linReg.predict(value)
    st.write("Результат линейной регрессии c нормализацией L2:", round((float(linReg_result)),3),'$')
    baggingReg_result=baggingReg.predict(value)
    st.write("Результат BaggingRegressor:",round(float(baggingReg_result),3))
    gradientBoostingReg_result=gradientBoostingReg.predict(value)
    st.write("Результат GradientBoostingRegressor:",round(float(gradientBoostingReg_result),3),'$')
    stackingReg_result=stackingReg.predict(value)
    st.write("Результат StackingRegressor:",round(float(stackingReg_result),3),'$')
    neoroReg_result=neoroReg.predict(value)
    st.write("Результат нейронной сети:",round(float(neoroReg_result),3),'$')



st.header("Загрузить свой датасет для обработки")
uploaded_file = st.file_uploader("Выберите файл в формате .csv", type='csv')
if uploaded_file:
    dataframe = pd.read_csv(uploaded_file)
    if 'price' in dataframe.columns:
        dataframe.pop('price')
        dataframe.pop('Unnamed: 0')
    getPredButton1=st.button("Получить предсказание при помощи линейной регрессии")
    getPredButton2=st.button("Получить предсказание при помощи BaggingRegressor")
    getPredButton3=st.button("Получить предсказание при помощи GradientBoostingRegressor")
    getPredButton4=st.button("Получить предсказание при помощи StackingRegressor")
    getPredButton5=st.button("Получить предсказание при помощи нейронной сети")
    if getPredButton1:
        linReg_result=linReg.predict(dataframe)
        st.write('Результат линейной регрессии:', pd.DataFrame(linReg_result, columns=["predicted_price"]))
    if getPredButton2:
        baggingReg_result=baggingReg.predict(dataframe)
        st.write("Результат BaggingRegressor:", pd.DataFrame(baggingReg_result,columns=['predicted_price']))
    if getPredButton3:
        gradientBoostingReg_result=gradientBoostingReg.predict(dataframe)
        st.write("Результат GradientBoostingRegressor:", pd.DataFrame(gradientBoostingReg_result,columns=['predicted_price']))
    if getPredButton4:
        stackingReg_result=stackingReg.predict(dataframe)
        st.write("Результат StackingRegressor:", pd.DataFrame(stackingReg_result,columns=['predicted_price']))
    if getPredButton5:
        neoroReg_result=neoroReg.predict(dataframe)
        st.write("Результат нейронной сети:", pd.DataFrame(neoroReg_result,columns=['predicted_price']))

