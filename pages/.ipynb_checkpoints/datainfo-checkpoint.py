import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data\diamonds.csv')



st.header('Dataset info')
st.write(data[:4])
st.text('''
    price
    Цена в долларах (\$326--\$18,823)
    carat
    Вес алмаза  (0.2--5.01)
    cut
    Качество разреза (Fair, Good, Very Good, Premium, Ideal)
    color
    Цвет алмаза, от J (худший) до D (лучший)
    clarity
    Оценка прозрачности алмаза (I1 (худший), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (лучший))
    x
    Длина в мм (0--10.74)
    y
    Ширина в мм (0--58.9)
    z
    Высота в мм (0--31.8)
    depth
    Процент отношения высоты от других измерений = 2 * z / (x + y) (43--79)
    table
    Отношение ширины в верхней точке к максимальной, в процентах (43--95)''')


   