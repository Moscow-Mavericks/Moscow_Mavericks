#!/usr/bin/env python
# coding: utf-8

# # Fundamentos de Análisis de Datos - Reservas de Hoteles

# ## 1. Control de Versiones

# Con el objetivo de organizarnos mejor y llevar un control de versiones a lo largo del proyecto, previniendo así cualquier error o complicación que se nos pueda presentar, se ha utilizado github, donde hemos creado un [repositorio] para el equipo.
# 
# [repositorio]: https://github.com/Moscow-Mavericks/Moscow_Mavericks/tree/develop

# ## 2. Definición de Objetivos

# El proyecto se basa en explorar el conjunto de datos de "**hotels**", el cual representa las reservas de un hotel de ciudad y de un hotel turístico con información como la fecha de la reserva, la duración de la estancia, el número de adultos, niños y/o bebés y el número de plazas de aparcamiento disponibles, entre otras variables de importancia en el ámbito.
# 
# A partir de está exploración, el objetivo es realizar una **regresión lineal** de tal forma que se pueda predecir el **número de noches reservadas**. Cabe destacar que principalmente se pensó en centrarnos en la variable **is_canceled** para predecir cuando una reserva sería cancelada, pero según se fue avanzando en el proyecto se vió más oportuno dejar esa variable para una futura regresión logística.
# 
# Resumen de objetivos:
# * Analizar el conjunto de datos **hotels**.
# * Realizar tratamiento de los **datos faltantes** de dicho conjunto de datos.
# * **Tratar las variablas** (tanto cualitativas como cuantitativas) de forma adecuada para su futuro uso en el modelo de regresión lineal.
# * **Seleccionar las variables** más oportunas para obtener un modelo óptimo.
# * Crear un modelo de **regresión lineal** centrándonos en el número de noches a reservar.
# * Realizar **diagnosis** basándonos en dicho modelo.
# 
# 
# * **Todos los objetivos a su vez se centran en aplicar los conocimientos vistos durante esta asignatura para asimilarlos y comprenderlos de la mejor manera posible.**

# ## 3. Análisis Exploratorio

# En primer lugar realizaremos un analisis exploratorio de  nuestra base de datos, con ello buscamos explorar, describir, resumir y visualizar la naturaleza de los datos recogidos en las variables del proyecto o investigación de interés, mediante la aplicación de técnicas simples de resumen de datos y métodos gráficos sin tomar asunciones para su interpretación.

# ### 3.1 Librerias
# 

# In[1]:


import pandas as pd
import numpy as np
import pandas_profiling
import io
import warnings
import matplotlib.pyplot as plt
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from missingpy import KNNImputer


# ### 3.2 Datos
# 

# Realizaremos la lectura de los datos haciendo uso de la libreria Pandas, la cual recoge las mayoria de funcionalidades necesarias durante esta etapa del proyecto. Además realizaremos un hed para comproibar que los datos se encuentran correctamente leidos.

# In[2]:


data = pd.read_csv('simple-hotels.csv')
data.head()


# Pasamos a mostrar los tipos, buscando posibles irregularidades en la lectura que sean fácilmente apreciables

# In[3]:


data.info()


# Vamos a observar un resumen sobre las variables numéricas:

# In[4]:


data.describe()


# Mostramos la matriz de correlación entre las variables. Posteriormente en la selección de variables se volverá a mostrar para realizar una mejor selección.

# In[5]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# Juntamos las columnas de año, mes y dia para crear un único datetime.
# 
# También añadimos una columna con el mes de forma numérica para usarlas posteriormente.

# In[6]:


data['arrival_datetime'] = pd.to_datetime(data['arrival_date_year'].map(str) + "-" + data['arrival_date_month'].map(str) + "-" + data['arrival_date_day_of_month'].map(str))
data['arrival_month'] = data['arrival_datetime'].map(str).str[5:7]


# In[7]:


data = data.sort_values(by=['arrival_datetime'], ascending=True)
data.head()


# Agrupamos los datos por la fecha total, y por año y mes para estudiar los cambios a lo largo del tiempo.

# In[8]:


data_groupby_year_month = data.groupby(by=["arrival_date_year", "arrival_date_month"])
data_groupby_date = data.groupby(by=["arrival_datetime"])


# #### 3.2.1 Hotel
# Tipo de hotel.
# Primero comenzamos observando la columna de tipo de hotel.
# 
# 

# In[9]:


data['hotel'].describe()


# Como vemos, solo hay dos tipos distintos de hoteles, y el más repetido es el "City Hotel".
# 
# 

# In[10]:


data['hotel'].hist()


# ##### Tabla de contingencias

# In[11]:


pd.crosstab(data['hotel'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# Creamos una tabla de contiengencias observando los porcentajes, de tal forma que se puede ver que el porcentaje de cancelados es mayor en "City Hotel".
# 

# ##### Evolución Histórica
# Ahora vamos a obtener columnas númericas a partir de la columna "hotel" (**one hot encoder**) para poder agrupar cuantos hoteles se reservan cada mes.
# 

# In[12]:


data_hotel_and_date = data[['arrival_date_year', 'arrival_month', 'hotel']]
data_hotel_and_date = pd.get_dummies(data_hotel_and_date, columns=["hotel"])[['arrival_date_year', 'arrival_month', 'hotel_City Hotel', 'hotel_Resort Hotel']]
data_hotel_and_date = data_hotel_and_date.groupby(by=['arrival_date_year', 'arrival_month']).sum()
data_hotel_and_date


# In[13]:


plt.figure(figsize=(500,10));
data_hotel_and_date.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.2 Is canceled
# 
# Se trata de unas de las variables más importantes, pues sería interesante tratar de predecirla en un futuro. En ella se recoge si la reserva ha sido o no cancelada.

# In[14]:


data['is_canceled'].describe()


# Haciendo uso de un histograma, comprobaremos la masa de cada una de las posibles categorias de nuestra variable objetivo, en este caso, podemos observar que en nuestro sample de 10k elementos, la variable no es equitativa, esto es factor a tener en cuenta en un futuro entrenamiento de modelo, puesto que son datos desbalanceados

# In[15]:


data['is_canceled'].hist()


# ##### Evolución histórica

# Pasamos a comprobar si existe alguna posible relación con el año, buscando posibles incidencias que nos permitan vislumbrar comportamientos anómalos

# In[16]:


data_canceled_and_date = data[['arrival_date_year', 'arrival_month', 'is_canceled']].groupby(by=['arrival_date_year', 'arrival_month']).sum()
data_canceled_and_date.head()


# Podemos observar que las cancelaciones parecen seguir un ciclo temporal, siendo mayores en las etapas veraniegas frente a las invernales. Esta relación puede constituir un síntoma de importancia a la hora de la predicción.

# In[17]:


plt.figure(figsize=(500,10));
data_canceled_and_date.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.3 Lead time 	
# 
# Se trata de el número de días que han pasado desde que se realizó la reserva hasta el momento en que empieza la reserva. Podemos observar que es una de las pocas variables cuantitativas continuas que posee nuestro dataset, lo que la convierte en una variable muy destacada a la hora de su tratamiento

# In[18]:


data['lead_time'].describe()


# Podemos observar que la distribución de la variable parece ajustarse a una distribucion exponencial, lo que convierte al logaritmo en su posible transformación natural. Esto será considerado en el apartado de tratamiento de variables cuantitativas

# In[19]:


data['lead_time'].hist()


# ##### Evolución histórica
# Vamos a mostrar como ha ido evolucionando la variable a lo largo de cada mes.

# In[20]:


data_lead_time_and_date = data[['arrival_date_year', 'arrival_month', 'lead_time']].groupby(by=['arrival_date_year', 'arrival_month']).sum()
data_lead_time_and_date.head()


# Del mismo modo que en el caso anterior, con la graficación de la variable frente al tiempo, podemos observar que la variable se encuentra fuertemente ligada a la estacionalidad.

# In[21]:


plt.figure(figsize=(500,10));
data_lead_time_and_date.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.4 Arrival date year 	

# Esta variable recoge el año de la llegada a la reserva. Esto nos permite realizar agrupaciones anuales de forma mucho más sencilla. Del mismo modo que con la variable a predecir, observamos que no se encuentra equitativamente distribuido nuestro sample, siendo mucho mas numerosos los registros del año 2016 y 2017

# In[22]:


data['arrival_date_year'].value_counts()


# Representamos el histograma para comprender las magnitudes de la desproporcionalidad de forma mas precisa

# In[23]:


plt.xticks(rotation=45)
data['arrival_date_year'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del año y pintarlo en una gráfica.

# In[24]:


data_prob_canceled_by_year = pd.crosstab(data['arrival_date_year'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_year


# In[25]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_year.plot.bar(figsize=(30,8));
plt.tight_layout()


# ##### Comparación con is_canceled
# Vamos a agrupar por año para ver como han ido cambiando las cancelaciones a lo largo de los años.

# In[26]:


data_year_and_canceled = data[['arrival_date_year', 'is_canceled']]
data_year_and_canceled = data_year_and_canceled.groupby(by=['arrival_date_year']).sum()
data_year_and_canceled


# In[27]:


plt.figure(figsize=(500,10));
data_year_and_canceled.plot(figsize=(30,8));
plt.tight_layout()


# #### 3.2.5 Arrival month 	
# Usamos arrival_month(numérica), en lugar de arrival_date_month(categórica), pues son equivalentes.

# In[28]:


data['arrival_month'].describe()


# In[29]:


data['arrival_month'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del mes y pintarlo en una gráfica.

# In[30]:


data_prob_canceled_by_month = pd.crosstab(data['arrival_month'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_month


# Se puede observar que las cancelaciones tambien se encuentran ligadas a la temporalidad mensual. Además, como es lógico, las gráficas son simetricas y opuestas, sumando 1.

# In[31]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_month.plot(figsize=(30,8));
plt.tight_layout()


# #### Comparación con is_canceled
# Vamos a agrupar por mes para ver como han ido cambiando las cancelaciones a lo largo de los meses.

# In[32]:


data_month_and_canceled = data[['arrival_month', 'is_canceled']]
data_month_and_canceled = data_month_and_canceled.groupby(by=['arrival_month']).sum()
data_month_and_canceled


# De nuevo podemos observar que el numero de cancelaciones en los meses de verano es mayor.

# In[33]:


plt.figure(figsize=(500,10));
data_month_and_canceled.plot(figsize=(30,8));
plt.tight_layout()


# #### 3.2.6 Arrival date week number 	

# La variable Arrival date week number almacena la semana del año en la que se realizó la entrada al hotel, Es información intrinseca de otras columas (Concretamente las de Date)

# In[34]:


data['arrival_date_week_number'].describe()


# In[35]:


data['arrival_date_week_number'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función de la semana y pintarlo en una gráfica.

# In[36]:


data_prob_canceled_by_weeknumber = pd.crosstab(data['arrival_date_week_number'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_weeknumber.head()


# In[37]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_weeknumber.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### Comparación con is_canceled
# Vamos a agrupar por el número de la semana para ver como han ido cambiando las cancelaciones a lo largo de las mismas.

# In[38]:


data_week_and_canceled = data[['arrival_date_week_number', 'is_canceled']]
data_week_and_canceled = data_week_and_canceled.groupby(by=['arrival_date_week_number']).sum()
data_week_and_canceled.head()


# In[39]:


plt.figure(figsize=(500,10));
data_week_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.7 Arrival date day of month 	

# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del día del mes y pintarlo en una gráfica.

# In[40]:


data_prob_canceled_by_daynumber = pd.crosstab(data['arrival_date_day_of_month'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_daynumber.head()


# In[41]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_daynumber.plot.bar(figsize=(30,8));
plt.tight_layout()


# ##### Comparación con is_canceled
# Vamos a agrupar por día del mes para ver como han ido cambiando las cancelaciones a lo largo de los mismos.

# In[42]:


data_day_and_canceled = data[['arrival_date_day_of_month', 'is_canceled']]
data_day_and_canceled = data_day_and_canceled.groupby(by=['arrival_date_day_of_month']).sum()
data_day_and_canceled.head()


# In[43]:


plt.figure(figsize=(500,10));
data_day_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.8 Stays in weekend nights 	
# Noches de fin de semana reservadas.

# In[44]:


data['stays_in_weekend_nights'].describe()


# In[45]:


data['stays_in_weekend_nights'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de noches de fin de semana y pintarlo en una gráfica.

# In[46]:


data_prob_canceled_by_weekend_nights = pd.crosstab(data['stays_in_weekend_nights'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_weekend_nights


# Aquí se pueden ver algunos detalles significativos. Cuando se reservan 10 noches de fin de semana NADIE ha cancelado nunca la reserva. Y cuando se reservan 7 noches de fin de semana, TODO el mundo ha cancelado la reserva.

# In[47]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_weekend_nights.plot.bar(figsize=(30,8));
plt.tight_layout()


# ##### Comparación con is_canceled
# Vamos a agrupar por la variable para ver como han ido cambiando las cancelaciones en función de la misma.

# In[48]:


data_weekend_nights_and_canceled = data[['stays_in_weekend_nights', 'is_canceled']]
data_weekend_nights_and_canceled = data_weekend_nights_and_canceled.groupby(by=['stays_in_weekend_nights']).sum()
data_weekend_nights_and_canceled


# Al ver estos resultado se aprecia que la probabilidad de antes es engañosa, ya que se debe a muestras muy pequeñas o incluso vacías.

# In[49]:


plt.figure(figsize=(500,10));
data_weekend_nights_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.9 Stays in week nights 	
# Noches entre semana reservadas.

# In[50]:


data['stays_in_week_nights'].describe()


# In[51]:


data['stays_in_week_nights'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de noches entre semana y pintarlo en una gráfica.

# In[52]:


data_prob_canceled_by_week_nights = pd.crosstab(data['stays_in_week_nights'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_week_nights.head()


# In[53]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_week_nights.plot.bar(figsize=(30,8));
plt.tight_layout()


# ##### Comparación con is_canceled
# Vamos a agrupar por la variable para ver como han ido cambiando las cancelaciones a lo largo de la misma.

# In[54]:


data_week_nights_and_canceled = data[['stays_in_week_nights', 'is_canceled']]
data_week_nights_and_canceled = data_week_nights_and_canceled.groupby(by=['stays_in_week_nights']).sum()
data_week_nights_and_canceled.head()


# In[55]:


plt.figure(figsize=(500,10));
data_week_nights_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.10 Adults 	
# 
# La variable adulst almacena el numero de adultos para el cual se realizó la reserva. Es una variable discreta, y que puede ser de importancia en proximas etapas, ya que podria estar relacionada con las cancelaciones, pues podria parecer más dificil realizar una cancelación de una reserva de un mayor número de asistentes.

# In[56]:


data['adults'].describe()


# En el gráfico sigueinte s epuede observar que la amplia mayoría de las reservas se encuentran adjudicadas a dos adultos, siendo el segundo valor mas repetido el 1 y residuales otros valores como pudiera ser el 20 o el 27

# In[57]:


data.groupby('adults')['babies'].count().plot(kind='bar')


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de adultos y pintarlo en una gráfica.

# In[58]:


data_prob_canceled_by_adults = pd.crosstab(data['adults'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_adults


# In[59]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_adults.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### Comparación con is_canceled
# Vamos a agrupar por el número de adultos para ver como han ido cambiando las cancelaciones en función de los mismos.

# In[60]:


data_adults_and_canceled = data[['adults', 'is_canceled']]
data_adults_and_canceled = data_adults_and_canceled.groupby(by=['adults']).sum()
data_adults_and_canceled


# In[61]:


plt.figure(figsize=(500,10));
data_adults_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.11 Children 	

# In[62]:


data['children'].describe()


# In[63]:


data['children'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de niños y pintarlo en una gráfica.

# In[64]:


data_prob_canceled_by_children = pd.crosstab(data['children'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_children


# In[65]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_children.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### Comparación con is_canceled
# Vamos a agrupar por el número de niños para ver como han ido cambiando las cancelaciones en función de los mismos.

# In[66]:


data_children_and_canceled = data[['children', 'is_canceled']]
data_children_and_canceled = data_children_and_canceled.groupby(by=['children']).sum()
data_children_and_canceled


# In[67]:


plt.figure(figsize=(500,10));
data_children_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.12 Babies

# In[68]:


data['babies'].describe()


# In[69]:


data['babies'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del nñumero de bebés y pintarlo en una gráfica.

# In[70]:


data_prob_canceled_by_babies = pd.crosstab(data['babies'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_babies


# In[71]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_babies.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### Comparación con is_canceled
# Vamos a agrupar por el número de bebés para ver como han ido cambiando las cancelaciones en función de los mismos.

# In[72]:


data_babies_and_canceled = data[['babies', 'is_canceled']]
data_babies_and_canceled = data_babies_and_canceled.groupby(by=['babies']).sum()
data_babies_and_canceled


# In[73]:


plt.figure(figsize=(500,10));
data_babies_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()


# #### 3.2.13 Meal
# Tipo de comida reservada.

# Como podemos observar la variable info es una variable categorica con 5 valores distintos, que representan lo siguiente:
# 
# * BB: Bed & Breakfast
# 
# * HB: Half Board (Breakfast and Dinner normally)
# 
# * FB: Full Board (Beakfast, Lunch and Dinner)
# 
# * SC: Self Catering
# 
# * Undefined
# 
# Podemos observar que hay valores faltyas, para una posibnle posterior imputación

# In[74]:


data.meal.describe()


# In[75]:


data.meal.value_counts()


# In[76]:


data.meal.hist()


# In[77]:


ax = sns.stripplot(y="stays_in_weekend_nights", x="meal",hue="is_canceled", data=data)


# #### 3.2.14 Country

# La variable country representa el país de la reserva, de modo que constituye una variable categórica con 108 valores diferentes. El país mas representado es Portugal.

# In[78]:


data.country.describe()


# In[79]:


data.country.value_counts().head()


# Vemos que el país del cual más información poseemos es portugal, seguido de Reino Unido y Francia. Analizaremos el número de cancelaciones por país de los 5 más abundantes

# In[80]:


data_topcountrys= data[data.country=='PRT']
data_topcountrys= data_topcountrys.append(data[data.country=='GBR'])
data_topcountrys= data_topcountrys.append(data[data.country=='FRA'])
data_topcountrys= data_topcountrys.append(data[data.country=='ESP'])
data_topcountrys= data_topcountrys.append(data[data.country=='DEU'])
data_prob_canceled_by_country = pd.crosstab(data_topcountrys['country'], data_topcountrys['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_country


# In[81]:


data_prob_canceled_by_country.plot.bar(figsize=(30,8))


# Podemos observar que el ratio de cancelaciones en portugal es bastante mas alto al del resto de los paises top de nuestro dataset.

# #### 3.2.15 Market segment

# La variable market segment representa el segmento de mercado que ha realizado la contratacion. Es una varibale categorica con 7 valores distintos.
# El término "TA" significa "Agentes de Viaje" y "TO" significa "Operadores Turísticos"

# In[82]:


data.market_segment.describe()


# In[83]:


data.market_segment.value_counts()


# In[84]:


data_prob_canceled_by_market = pd.crosstab(data['market_segment'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_market


# In[85]:


data_prob_canceled_by_market.plot.bar(figsize=(30,8))


# Podemos observar que las reservas de grupos son las mas propensas a cancelar las reservas, siendo esta la opcion mas habital. En el lado opuesto tenemos el caso de la aviación.

# #### 3.2.16 Distribution channel

# Canal de distribución de reservas. El término "TA" significa "Agentes de Viaje" y "TO" significa "Operadores Turísticos"

# In[86]:


data.distribution_channel.describe()


# In[87]:


data.distribution_channel.value_counts()


# In[88]:


data_prob_canceled_by_market = pd.crosstab(data['distribution_channel'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_market


# In[89]:


data_prob_canceled_by_market.plot.bar(figsize=(30,8))


# Podemos observar que las reservas por TA/TO suelen cancelarse más amenudo

# #### 3.2.17 Is repeated guest

# Valor que indica si el nombre de la reserva era de un huésped repetido (1) o no (0)

# In[90]:


data.is_repeated_guest.describe()


# In[91]:


data.is_repeated_guest.sum()


# In[92]:


data_repeted_guest=data[data.is_repeated_guest==1]
data_repeted_guest.is_canceled.sum()


# In[93]:


data_repeted_guest=data[data.is_repeated_guest==0]
data_repeted_guest.is_canceled.sum()


# Podemos observar que de los repetidor, solo 52 cancelaron su viaje, es decir, 16%. Frente a los no repetidores, entre los que la tasa de cancelacion es 38%

# #### 3.2.18 Previous cancellations

# Número de reservas anteriores que fueron canceladas por el cliente antes de la reserva actual

# In[94]:


data.previous_cancellations.describe()


# In[95]:


data.previous_cancellations.value_counts()


# In[96]:


data_prob_canceled = pd.crosstab(data['previous_cancellations'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled


# In[97]:


data_prob_canceled.plot.bar(figsize=(30,8))


# Podemos observar, que la tónica general es no tener cancelaciones previas. Pero a mayor número de cancelaciones previas, se cancela menos la reserva.

# #### 3.2.19 Previous bookings not canceled

# Número de reservas anteriores no canceladas por el cliente antes de la reserva actual

# In[98]:


data.previous_bookings_not_canceled.describe()


# In[99]:


data_prob_ncanceled = pd.crosstab(data['previous_bookings_not_canceled'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_ncanceled.plot.bar(figsize=(30,8))


# Podemos observar, que el comportamiento es similar al caso anterior, es decir, aquellos con pocas cancelaciones previas, es mas probable que cancelen, pero a mayor numero de cancelaciones general, es mas dificil que sea cancelado.

# #### 3.2.20 Reserved room type

# Código de tipo de habitación reservada. El código se presenta en lugar de la designación por razones de anonimato.

# In[100]:


data.reserved_room_type.describe()


# In[101]:


data.reserved_room_type.value_counts()


# In[102]:


data_prob_res = pd.crosstab(data['reserved_room_type'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[103]:


data_prob_res.plot.bar(figsize=(30,8))


# Podemos observar que las cancelaciones son similares en todos los tipos de habitacion reservada, salvo en 3 casos llamativos:
# * Tipo H donde las cancelaciones son ligeramente superiores
# * Tipo L no existen cancelaciones
# * Tipo P solo existen cancelaciones

# #### 3.2.21 Assigned room type

# Código del tipo de habitación asignada a la reserva. A veces el tipo de habitación asignada difiere del tipo de habitación reservada debido a razones de operación del hotel (por ejemplo, sobreventa) o por petición del cliente. El código se presenta en lugar de la designación por razones de anonimato.

# In[104]:


data.assigned_room_type.describe()


# In[105]:


data.assigned_room_type.value_counts()


# Puede ser interesante si los tipos asignados son los tipos reservados, y la relación que ello tiene con la cancelación de la reserva

# #### 3.2.22 Booking changes

# Número de cambios/enmiendas hechos a la reserva desde el momento en que la reserva fue ingresada en el PMS hasta el momento del registro o cancelación.

# In[106]:


data.booking_changes.describe()


# In[107]:


data.booking_changes.value_counts()


# In[108]:


data_prob_res = pd.crosstab(data['booking_changes'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[109]:


data_prob_res.plot.bar(figsize=(30,8))


# Podemos destacar que las cancelaciones aumentan cuando sufren un mayor numero de cambios en las reservas, ademas para cambios pequeños, las reservas no se resienten en terminos de cancelación

# #### 3.2.23 Deposit type

# Indicación sobre si el cliente hizo un depósito para garantizar la reserva. Esta variable puede asumir tres categorías: No Deposito - no se hizo ningún depósito; No Reembolso - se hizo un depósito con un valor por debajo del costo total de la estancia.

# In[110]:


data.deposit_type.describe()


# In[111]:


data.deposit_type.value_counts()


# In[112]:


data_prob_res = pd.crosstab(data['deposit_type'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[113]:


data_prob_res.plot.bar(figsize=(30,8))


# Muy destacable que las reservas sin refound son las mas canceladas

# #### 3.2.24 Agent
# 

# La identificación de la agencia de viajes que hizo la reserva

# In[114]:


data.agent.describe()


# In[115]:


data_prob_res = pd.crosstab(data['agent'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res.head()


# Mucha cardinalidad en la variable, dado que hay 200+ valores de agencia y no tenemos traduccion de estos id (VAR CANDIDATA A DESAPARECER)

# #### 3.2.25 Company

# La variable company es un identificador de la empresa o entidad que realiza la reserva

# In[116]:


data.company.unique()
frecuencia_id_compañia=data.company.value_counts()
porcentaje_frecuencias_id_compañia=data.company.value_counts(40.0)
print("El número de reserva hechas por cada id:",frecuencia_id_compañia)
print("El porcentaje de reservas por cada id:", porcentaje_frecuencias_id_compañia)


# In[117]:


data_prob_res = pd.crosstab(data['company'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res.head()


# Mucha cardinalidad en la variable, dado que hay 140+ valores de company y no tenemos traduccion de estos id (VAR CANDIDATA A DESAPARECER)

# #### 3.2.26 Days in waiting list

# Variable cuantitativa que recoge el número de días que pasan hasta que el usuario confirma la reserva

# In[118]:


data["days_in_waiting_list"].describe()


# In[119]:


data["days_in_waiting_list"].hist(width=25)


# Comparación respecto a la variable is_canceled:

# In[120]:


data_prob_res=pd.crosstab(data['days_in_waiting_list'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res.head()


# In[121]:


data["days_in_waiting_list"].median()


# Observamos que en esta variable la desviación típica podría indicar cierta variabilidad pero observando la mediana no se observa tal, por lo que esta se explica por algunos valores anómalos. Lo que observamos es que esta variable no se desvía mucho de cero, lo que nos indica que no tiene mucho potencial explicativo.

# #### 3.2.27 Customer type

# Variable categórica que clasifica los tipos de reservas en: contractuales, grupales, transitorias o perteneciente a una transitoria:
# 
# * Las reservas contractuales son las que tienen una asignación o a algún otro tipo de contrato.
# 
# * Las grupales están asociadas a grupos.
# 
# * Las transitotorias son las que no están recogidas en ninguna de las categorías anteriores, siendo las pertennecientes a una transitoria (Trasient-party) una reserva asociada a una transitoria (Transient).

# In[122]:


data["customer_type"].unique()


# In[123]:


data["customer_type"].hist()


# Comparación respecto a la variable is_canceled:

# In[124]:


data_prob_res=pd.crosstab(data['customer_type'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[125]:


data_prob_res.plot.bar(figsize=(30,8))


# Se puede observar que las reservas grupales son las que menos sufren cancelaciones, siendo las transitorias las más canceladas.

# #### 3.2.28 ADR

# Variable cuantitativa que refleja el ratio entre precio pagado por la reserva o estancia, dividido entre el número de noches de estancia del usuario

# In[126]:


data["adr"].describe()


# In[127]:


data["adr"].hist()


# In[128]:


pd.crosstab(data['adr'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# En esta variable observamos que es similar a una distribución exponencial, por lo que sería recomendable transformarla para que su distribución se asemeje más a una normal.

# #### 3.2.29 Required car parking spaces

# Número de plazas de parking solicitadas por el usuario

# In[129]:


data["required_car_parking_spaces"].describe()


# In[130]:


data["required_car_parking_spaces"].hist()


# Comparación respecto a la variable is_canceled:

# In[131]:


data_prob_res=pd.crosstab(data['required_car_parking_spaces'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[132]:


data_prob_res.plot.bar(figsize=(30,8))


# En esta variable observamos que las reservas sin petición de plaza d eaparcamiento suelen cancelarse, mientras que las que lo solicitan no se cancelan.

# #### 3.2.30 Total of special request

# Número total de requisitos especiales solicitados por el usuario.

# In[133]:


data["total_of_special_requests"].describe()


# In[134]:


data["total_of_special_requests"].hist()


# En esta variable se puede observar una distribución de probabilidad similar a la exponencial, por lo que en el caso de incluirla en el modelo de regresión sería recomendable su transformación.

# Comparación respecto is_canceled:

# In[135]:


data_prob_res=pd.crosstab(data['total_of_special_requests'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[136]:


data_prob_res.plot.bar(figsize=(30,8))


# Observamos que las reservas sin ninguna petición especial suelen cancelarse en mayor número, las reservas con 5 peticiones especiales no se cancelan.

# #### 3.2.31 Reservation status

# Último estado de la reserva, variable categórica que clasifica en: cancelado, check-out (finalizada estancia), sin aparecer (el usuario no ha realizado el check in, informando al hotel la razón)

# In[137]:


data["reservation_status"].describe()


# In[138]:


data["reservation_status"].hist()


# Comparación respecto la variable is_canceled:

# In[139]:


pd.crosstab(data['reservation_status'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# Esta es una variable cualitativa que se refiere la misma información recogida en la variable "is_canceled", por lo que se descarta para su utilización como variable predictora al ser potencialmente colineal con la variable anteriormente mencionada.

# #### 3.2.32 Reservation status date

# Fecha de la última actualización del estado de la reserva

# In[140]:


data["reservation_status_date"].describe()


# In[141]:


fechas=pd.to_datetime(data["reservation_status_date"])
fechas.hist()


# Esta variable consideramos que va a aportar poco a la explicabilidad del modelo de regresión para predecir el número de noches de hospedaje, por lo que es probable que no sea considerada.

# ## 4. Procesado de variables cualitativas

# Procesamos las variables cualitativas de tal forma que estas pasen a ser variables/indicadores dummies.

# In[142]:


data_processed = pd.get_dummies(data, columns=["hotel", "meal", "deposit_type"],drop_first=True)


# Además, vamos a sintetizar las columnas  "reserved_room_type", "assigned_room_type", en una única feature booleana que sea afirmativa en caso de que la habitación reservada fuera la misma a la asignada y viceversa. Además, eliminaremos la variable "reservation_status", pues su contenido puede ser transcrito de otras variables como is canceled.

# In[143]:


pd.set_option("display.max_rows", 20, "display.max_columns", None)
data_processed["reserverd/assigned"]= np.where(data_processed["reserved_room_type"]==data_processed["assigned_room_type"],1 , 0)
data_processed=data_processed.drop(["market_segment", "distribution_channel","reserved_room_type","assigned_room_type","reservation_status"],axis=1)


# Por último, vamos a tratar de manera especial la variable "customer_type", puesto que existen dos categorias de esta variable que se encuentran altamente relacionadas, concretamenta esta relación es de contenido entre las categorias Transistent y Transistent-Party. Dado que una categoria se encuentra embebida en otra, vamos a unirlas en una única

# In[144]:


data_processed["customer_type"]= np.where(data_processed["customer_type"]=='Transient','Transient-Party',data_processed["customer_type"])
data_processed = pd.get_dummies(data_processed, columns=["customer_type"],drop_first=True)


# ## 5. Transformación de variables cuantitativas

# Las variables de entrada numéricas pueden tener una distribución muy sesgada o no estándar. Esto podría ser causado por valores atípicos en los datos, distribuciones multimodales, distribuciones altamente exponenciales y más.
# 
# Muchos algoritmos de aprendizaje automático prefieren o funcionan mejor cuando las variables de entrada numéricas tienen una distribución de probabilidad estándar.
# 
# Seleccionaremos para la transformación aquellas variables que son cuantitativas no categóricas, en nuestro caso dos, 'lead_time','adr'.

# In[145]:


numerical_df = data_processed
numerical_df = numerical_df.select_dtypes(np.number)
numerical_df = numerical_df[['lead_time','adr']]
numerical_df.hist()


# Para transformar las distribuciones anteriores, utilizaremos el metodo Power Transformer, con ello, buscamos mejorar la performance de futuros modelos. Las transformaciones de potencia son una familia de transformaciones paramétricas y monótonas que tienen por objeto mapear los datos de cualquier distribución lo más cerca posible de una distribución gaussiana a fin de estabilizar la varianza y reducir al mínimo la asimetría.
# 
# El transformador de potencia proporciona actualmente dos de esas transformaciones de potencia, la transformación de Yeo-Johnson y la de Box-Cox. Usaremos la transformación Yeo-Johnson, pues tenemos valores incomplatibles con Box-Cox (Valores no estrictamente positivos).

# In[146]:


#yeo-johnson
from sklearn import preprocessing
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
numerical_df_yeo = pt.fit_transform(numerical_df)
res = pd.DataFrame(numerical_df_yeo,columns=['lead_time_trans','adr_trans'])
data_processed[['lead_time','adr']]=res
res.hist()


# ## 6. Detección, tratamiento e imputación de datos faltantes
# 
# Debido a que nuestro dataset no tiene muchos datos faltantes, y gracias al análisis previo, se han estudiado variables sin datos faltantes pero con datos ilógicos que podrían ser tratados como tal.
# 
# Para tratar con los datos faltantes vamos a usar las librerías IterativeImputer y KNNImputer. Primero vamos a contar cuantos "null" hay en total.

# In[147]:


data.isnull().sum()


# Podemos ver que solo hay tres variables con valores "null". Pero vamos a ver otras variables que pueden contener datos faltantes aunque no tengan "null".
# 
# Precisamente las variables en las que hay valores "null", no nos valen. Ya que Country es una variable categórica y Agent y Company son identificadores únicos. Sin embargo, hay otras variables sin valores "null" pero de las que se pueden obtener datos faltantes (como 0 por ejemplo) en función de lo que significa cada variable.

# ### 6.1 Lead time
# Esta variable mide el número de días que transcurrieron entre la fecha de entrada de la reserva y la fecha de llegada por lo que se puede considerar como dato faltante si se tiene el valor 0. Lo haremos así para tener más datos faltantes.

# In[148]:


data_processed[data_processed['lead_time'] == 0].head()


# ### 6.2 Adults
#  
# Esta variable mide el número de adultos en la reserva, por lo que se puede considerar como dato faltante si se tiene el valor 0, ya que entendemos que solo se pueden realizar reservas por adultos (para así tener más datos faltantes).

# In[149]:


data_processed[data_processed['adults'] == 0][['adults', 'children', 'babies']].head()


# #### Iterative Imputer
# Vamos a imputar los datos faltantes, teniendo en cuenta solo las variables Adults, Children y Babies, ya que después se hará con todo el dataframe.

# In[150]:


data_adults = data_processed[['adults', 'children', 'babies']]


# In[151]:


imp = IterativeImputer(max_iter = 10, random_state = 0)
data_adults = imp.fit_transform(data_adults)
data_adults = pd.DataFrame(data=data_adults, columns=['adults', 'children', 'babies'])
data_adults.head()


# #### KNNImputer

# In[152]:


data_adults = data_processed[['adults', 'children', 'babies']]
imp = KNNImputer(n_neighbors=4, weights="uniform")
data_adults = imp.fit_transform(data_adults)
data_adults = pd.DataFrame(data=data_adults, columns=['adults', 'children', 'babies'])
data_adults.head()


# ### 6.3 All Dataset
# 
# ##### Iterative Imputer
# 
# Por último, realizamos la imputacion de los datos faltantes con todas las columnas, para ver el resultado. Quitamos las variables del tipo string para poder realizar el proceso.

# In[153]:


data_imputer = data_processed.drop(['country','arrival_date_month', 'reservation_status_date', 'arrival_datetime','agent','company'], axis=1)
data_imputer['adults'] = data_imputer['adults'].replace(0, np.nan)
data_imputer['lead_time'] = data_imputer['lead_time'].replace(0, np.nan)


# In[154]:


imp = IterativeImputer(max_iter = 3, random_state = 0,verbose=1)
data_iterative_imputer = imp.fit(data_imputer)
data_iterative_imputer = imp.transform(data_imputer)


# In[155]:


df_imputer=pd.DataFrame(data_iterative_imputer,columns=data_imputer.columns)
df_imputer.isna().sum()


# Como podemos observar las variables con valores faltantes han sido imputadas, vamos a realizar un pequeño análisis de cambio de los principales estadisticos de nuestro DataFrame

# In[156]:


data_imputer['adults'].describe()


# In[157]:


df_imputer['adults'].describe()


# Podemos observar que la variación de los principales estadisticos es minima, señal del bnuen funcionamiento del método de imputacion utilizado.

# ## 7. Selección de variables

# ### Recapitulación
# Vamos a realizar un pequeño sumario de lo que tenemos a estas alturas en nuestro dataframe original, para ver el estado de las variables

# In[158]:


pd.set_option("display.max_rows", 20, "display.max_columns", None)
data_processed = df_imputer
data_processed


# ### Variable Objetivo
# Como podemos observar, nuestra variable a predecir se encuentra dividida en dos columnas, de modo que vamos a sintetizar informacion en una única columna, retirando las otras dos.

# In[159]:


data_processed['stays_nights']=data_processed.stays_in_weekend_nights+data_processed.stays_in_week_nights
data=data_processed.drop(['stays_in_weekend_nights','stays_in_week_nights'],axis=1)


# Además, vamos a retirar la variable arrival_date_week_number, puesto que con el día, mes y año es más que suficiente para la identificacion de la fecha de la reserva

# In[160]:


data_processed=data_processed.drop(['arrival_date_week_number'],axis=1)


# Pasamos todas las columnas numéricas a entero para normalizar los tipos de las distintas variables.

# In[161]:


data=data.drop(['arrival_date_year'],axis=1)
columns_ints=['is_canceled','arrival_date_day_of_month', 'adults', 'children', 'babies',
       'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'booking_changes',
       'days_in_waiting_list', 'required_car_parking_spaces',
       'total_of_special_requests', 'arrival_month', 'hotel_Resort Hotel',
       'meal_FB', 'meal_HB', 'meal_SC', 'meal_Undefined',
       'deposit_type_Non Refund', 'deposit_type_Refundable',
       'reserverd/assigned', 'customer_type_Group',
       'customer_type_Transient-Party', 'stays_nights']
data[columns_ints] = data[columns_ints].astype(int)


# ### Train y Test
# 
# Separamos entre variables predictoras y predecida en primer lugar, seguidamente, realizaremos particiones para entrenar y validar el modelo, pàra ello, haremos uso de la función de sklearn de model_selection

# In[162]:


Y = data.stays_nights
X = data.loc[:, data.columns != 'stays_nights']


# Vamos a dividir nuestro conjunto de datos en train y test, tanto para las posibles variables predictoras como para la varabiable  a predecir

# In[163]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)


# ### Variance Threshold
# 
# Procedemos a eliminar las variables con menor varianza usando Variance Threshold. Con ello se pretende la identificación de las variables que poseen menor varianza, para evitar su coinsideracióin en los modelos.

# In[164]:


from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0.05)
constant_filter.fit(X_train)


# In[165]:


constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[constant_filter.get_support()]]
print(len(constant_columns))


# Vemos que hay 7 variables con una varianza menor a 0.05, vamos a ver cuales son.

# In[166]:


for column in constant_columns:
    print(column)


# De estas variables creemos que "babies" y "is_repeated_guest" son detalles importantes aún teniendo una baja varianza, por lo que son candidatas a no ser seleccionadas para el entrenamiento de los modelos.

# ### Correlación de variables
# 
# Vamos a realizar un analisis de correlación para eliminar aquellas variables que se encuentren altamente correlacionadas.

# In[167]:


corr_matrix = X_train.corr(method="pearson")
corr_matrix.style.background_gradient(cmap='coolwarm')


# In[168]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

def select_features_corr(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_regression, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def select_features_mutual(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# Para seleccionar variables este método mide la dependencia entre la variables, para medir esta dependencia utiliza el método de los k-vecinos.
# 
# El método de los K-vecinos (K-neighborgs), en regresión, las entradas del algoritmo ***k*** es el número de observaciones por variable que el modelo utiliza para estimar la salida o predicción. Esto es, la predicción es la media de la variable observada ***k*** veces.
# 
# Los valores cercanos a cero indican que las variables son independientes, valores más altos señalan una dependecnia más alta.
# 
# Como se observa no hya ninguna varible que supere un valor de dependencia de 0.05, por lo que, es difícil justificar la dependencia de las variables predictoras con la variable a predecir.

# In[169]:


columns=X_train.columns
results_corr=dict()
X_train_fs, X_test_fs, fs = select_features_corr(X_train, y_train, X_test)
for i in range(len(columns)):
    results_corr[columns[i]]=fs.scores_[i]
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
results_corr= {k: v for k, v in sorted(results_corr.items(), key=lambda item: item[1])}


# In[170]:


X_train_fs, X_test_fs, fs = select_features_mutual(X_train, y_train, X_test)
results_mut=dict()
for i in range(len(columns)):
    results_mut[columns[i]]=fs.scores_[i]
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
results_mut= {k: v for k, v in sorted(results_mut.items(), key=lambda item: item[1])}


# ### Random Forest
# 
# Vamos a comprobar cuales son las variables que más aportan al modelo, para eliminar las menos importantes, para ello haremos uso de Random Forest.
# 
# Random Forest utiliza árboles de decisión de clasificación en submuestras variables del conjunto de datos y utiliza el promedio para mejorar la precisión de la predicción y controlar el exceso de ajuste. 
# El tamaño de la submuestra se controla con el parámetro max_samples si bootstrap=True (por defecto), de lo contrario se utiliza todo el conjunto de datos para construir cada árbol.
#  
# n_estimators: Es el numero de árboles
#  
# random_state: Controla tanto la aleatoriedad de las muestras utilizadas al construir los árboles (si bootstrap=True, si no coge todas las variables)
# como el muestreo de las características.

# In[171]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf.fit(X_train, y_train);


# In[172]:


feature_list = list(X_train.columns)

# Importances
importances = list(rf.feature_importances_)

# Tuplas variable - importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort importances
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# ## 8. Ajuste, interpretación y diagnosis del modelo de regresión lineal múltiple

# ### Modelos a planteados

# A la vista de los resultados anteriores, realizaremos 4 entrenamientos diferentes:
# 
# * El primero de ellos constará de todas las features
# 
# * El segundo las mejores features determinadas por f_regression
# 
# * El tercero las mejores features determinadas por mutual_info_regression
# 
# * El cuarto una mezcla de las mejores consideradas por ambos modelos
#     
# Por ultimo, evaluaremos un modelo de regresión LASSO (https://machinelearningmastery.com/lasso-regression-with-python/) considerando  10-fold cross-validation (https://scikit-learn.org/stable/modules/linear_model.html)

# In[173]:


import statsmodels.api as sm
import lmdiag

def launch_linear_model(input_data,y_train,selected_var=[],with_intercept=1,verbose=1):
    if len(selected_var)>0:
        X_train = input_data[selected_var]
    else:
         X_train = input_data
    if with_intercept==1:
        print('Intercept added')
        X_train = sm.add_constant(X_train)
    print("Training...")
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    if verbose==1:
        print(results.summary())
    if verbose==2:
        print(results.summary())
        print('Calculando plots...')
        lmdiag.plot(results)
    print('Done.')
        
    


# ### Modelo total_features

# En primer lugar entrenamos al modelo con los datos de train.

# In[174]:


launch_linear_model(X_train,y_train,verbose=0)


# ### Modelo corr_features

# Vamos a seleccionar las variables mas significativas seleccionadas por el modelo que hace uso de las correlaciones, y realizaremos la prueba del ajuste del modelo haciendo uso de esas features

# In[175]:


vars_selected=['reserverd/assigned','deposit_type_Non Refund','is_repeated_guest','adults','meal_HB',
                        'customer_type_Transient-Party','hotel_Resort Hotel']
launch_linear_model(X_train,y_train,vars_selected,verbose=0)


# ### Modelo lineal con features seleccionadas por LASSO 
# 
# LASSO es un algoritmo supervisado que identifica las variables que están fuertemente asociadas con la variable de respuesta.  Luego, Lasso fuerza los coeficientes de las variables hacia cero, a este proceso se le denomina proceso de contracción. Esto es para que el modelo sea menos sensible al nuevo conjunto de datos. Estos procesos ayudan a seleccionar menos variables de entrada.
# 

# In[176]:


from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
reg = LassoCV()
reg.fit(X_train, y_train)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X_train,y_train))
coef = pd.Series(reg.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(np.abs(coef) > 0.25)) + " variables and eliminated the other " +  str(sum(np.abs(coef) < 0.25)) + " variables")
imp_coef = coef.sort_values()
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[177]:


var_selected=list(coef[np.abs(coef)>0.25].index)
launch_linear_model(X_train,y_train,var_selected,verbose=0)


# ### Modelo lineal con features seleccionadas por BACKWARD ELIMINATION

# In[178]:


#Backward Elimination
cols = list(X_train.columns)
pmax = 0.5
while (len(cols)>0):
    p= []
    X_1 = X_train[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y_train,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.000005):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols


# In[179]:


launch_linear_model(X_train,y_train,selected_features_BE,verbose=0)


# ### Modelo mut_features

# Vamos a seleccionar las variables obtenidas como representativas haciendo uso del método de mutual information, correremos el modelo lineal y graficaremos sus residuos. Además, será ampliamente comentado debido a que es el modelo con el menor numero de variables, y al hecho de que todos los modelos presentados anteriormente poseen una distribución similar en residuos y comportamiento, y por tanto, siumilares conclusiones.

# In[180]:


vars_selected=['arrival_month','deposit_type_Non Refund','is_repeated_guest','reserverd/assigned','hotel_Resort Hotel']
launch_linear_model(X_train,y_train,vars_selected,verbose=1)


# In[181]:


from IPython.display import Image
Image(filename='plot1.png')


# Vamos a realizar de manera profunda la diagnosis del siguiente modelo, dado que los problemas que en el experimentamos son recurrentes a lo largo de las diferentres selecciones de variables que hemos realizado. Ademas, realizaremos la diagnosis haciendo uso de la intercepta, y eliminandola, debido al significativo cambio que el R^2 ajustado presenta.
# 
# En primer lugar, haciendo uso de la intercepta, podemos observar los siguiente:
# 
# * **R-squared**: 0.094  es una medida estadística que representa la proporción de la varianza de una variable dependiente que se explica por una o varias variables independientes en un modelo de regresión.
# 
# * **Adj. R-squared**: 0.093 nos permite comparar el poder explicativo de los modelos de regresión. Podemos observar que es muy bajo.
# 
# * **F-statistic**: 139.0 prueba la importancia general del modelo de regresión.  Específicamente, prueba la hipótesis nula de que todos los coeficientes de regresión son iguales a cero.  Esto prueba el modelo completo contra un modelo sin variables y con la estimación de la variable dependiente siendo la media de los valores de la variable dependiente.
# 
# * **Prob (F-statistic)**: 9.81e-141 es la probabilidad de que la hipótesis nula del modelo completo sea cierta (es decir, que todos los coeficientes de regresión sean cero). Este valor tan bajo implica que al menos algunos de los parámetros de regresión no son cero y que la ecuación de regresión tiene cierta validez para ajustar los datos.
# 
# Con respecto a los gráficos de residuos:
# 
# * **Residuals-Fitted**: Podemos observar heterocedasticidad en los residuos, es decir, la varianza de los errores no es constante en todas las observaciones realizadas. Esto implica el incumplimiento de una de las hipótesis básicas sobre las que se asienta el modelo de regresión lineal. De ella se deriva que los datos con los que se trabaja son heterogéneos, ya que provienen de distribuciones de probabilidad con distinta varianza.
# 
# * **Normal Q-Q**: Podemos observar que las colas se alejan de manera muy notable, de modo que los residuos no estan distribuidos de manera normal
# 
# * **Scale-Location**: Con este gráfico se corrobora que existe heterocedasticidad de los residuos
# 
# * **Residuals-Leverage**: Nos reporta nos numeros que son importantes en la gráfica, pero podemos observar que no visualizamos siquiera la distancia de Cook, de modo que ninguno de nuestros valores es influyente
# 

# In[182]:


launch_linear_model(X_train,y_train,vars_selected,with_intercept=0,verbose=1)


# In[183]:


Image(filename='plot2.png')


# Vamos a realizar la diagnosis del segundo modelo, haciendo uso de la intercepta y de las mismas variables que el caso anterior, debido al significativo cambio que el R^2 ajustado presenta.
# 
# Podemos observar los siguiente:
# 
# * **R-squared**:  0.670  Podemos observar que es bastante superior, lo que a priori significaría
# 
# * **Adj. R-squared**:0.669 podemos observar que es considerablemente mejor, mejorando el poder explicativo del modelo de regresión
# 
# * **F-statistic**: 2714 es mucho mayor, probando asi la importancia general del modelo de regresión a priori. 
# 
# * **Prob (F-statistic)**:  0.00 de nuevo, este valor tan bajo implica que al menos algunos de los parámetros de regresión no son cero y que la ecuación de regresión tiene cierta validez para ajustar los datos.
# 
# Con respecto a los gráficos de residuos, podemos observar las mismas cualidades que el análisis realizado anteriormente. Esto nos prueba, que aunque los estadisticos usuales de la regresion pudieran parecer mucho mejores, nuestro modelo sigue sin ser representativo para la prediccion den nuestra variable objetivo. Este caso constituye un buen ejemplo de como los estadisticos que usualmente utilizamos para la validación de modelos de regresión lineal, no siempre son válidos.

# In[ ]:




