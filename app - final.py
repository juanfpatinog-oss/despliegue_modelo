#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
#Cargamos el modelo
import pickle
filename = 'modelo-NN.pkl'
modelo, labelencoder, min_max_scaler, variables = pickle.load(open(filename, 'rb'))
 
 
#Se crea interfaz gráfica con streamlit para captura de los datos
import streamlit as st
st.title('Predicción de cluster de un material nuevo')
Unidad_Medida = st.selectbox('Unidad Medida', ["'UN'","'ML'","'KG'","'LT'"]
Numero_de_Transacciones = st.slider('Numero de Transacciones',min_value=1, max_value=150, value=20, step=1)
Cantidad = st.slider('Cantidad',min_value=-30000, max_value=0, value=20, step=1)
Reintegros = st.slider('Reintegro',min_value=0, max_value=2500, value=20, step=1)
Precio_Unitario = st.slider('Precio_Unitario',min_value=1, max_value=300000000, value=20, step=100000)
Entrega = st.selectbox('Unidad Medida', ["'TOTAL'","'PARCIAL'","'REINTEGRO'"]

 
#Dataframe
datos = [[Unidad_Medida, Numero_de_Transacciones, Cantidad, Reintegros, Precio_Unitario, Entrega]]
data = pd.DataFrame(datos, columns=['Unidad_Medida', 'Numero_de_Transacciones', 'Cantidad', 'Reintegros', 'Precio_Unitario', 'Entrega']) #Dataframe con los mismos nombres de variables

 
#Se realiza la preparación
data_preparada=data.copy()
#En despliegue drop_first= False
data_preparada = pd.get_dummies(data_preparada, columns=['Numero de Transacciones','Cantidad','Reintegros','Precio Unitario'], drop_first=False, dtype=int)
data_preparada.head()
#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)
data_preparada.head()
 
#Se normaliza las variables númericas para la Red
#En los despliegues no se llama fit
data_preparada[['Numero_de_Transacciones']]= min_max_scaler.transform(data_preparada[['Numero_de_Transacciones']])
data_preparada[['Cantidad']]= min_max_scaler.transform(data_preparada[['Cantidad']])
data_preparada[['Reintegros']]= min_max_scaler.transform(data_preparada[['Reintegros']])
data_preparada[['Precio_Unitario']]= min_max_scaler.transform(data_preparada[['Precio_Unitario']])
data_preparada.head()
 
#Hacemos la predicción
Y_pred = modelo.predict(data_preparada)
print(Y_pred)
data['Prediccion']=Y_pred
data.head()
 
#Mostramos la predicción
data
 