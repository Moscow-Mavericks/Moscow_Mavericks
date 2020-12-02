#!/usr/bin/env python
# coding: utf-8

# # Análisis y modelización

# ## Añadir heatmap de datos faltantes!!!!!!!!
# https://github.com/ResidentMario/missingno

# ## Carga de librerias
# 
# Para el correcto funcionamiento del análisis, se requiere el uso de las siguientes librerias:

# In[1]:


import pandas as pd
import pandas_profiling
import io
import warnings
import matplotlib.pyplot as plt
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
warnings.filterwarnings('ignore')


# ## Datos
# 
# ### Contexto
# 
# ¿Alguna vez te has preguntado cuándo es la mejor época del año para reservar una habitación de hotel? ¿O la duración óptima de la estancia para obtener la mejor tarifa diaria? ¿Y si quisiera predecir si un hotel es probable que reciba un número desproporcionadamente alto de solicitudes especiales?
# 
# Mediante este conjunto de datos de reservas de hotel nos proponemos explorar estas preguntas!
# 
#  ### Contenido
#  
# Este conjunto de datos contiene información sobre las reservas de un hotel de ciudad y de un hotel turístico, e incluye información como la fecha de la reserva, la duración de la estancia, el número de adultos, niños y/o bebés y el número de plazas de aparcamiento disponibles, entre otras cosas.
# 
# Toda la información de identificación personal se ha eliminado de los datos.
# 
# ### Carga de datos

# In[2]:


data = pd.read_csv('simple-hotels.csv')
data.head()


# In[3]:


data.info()


# Podemos observar que nuestro data.set consta de 31 variables, 20 variables numericas, aunque muchas de ellas son categóricas, y el resto de tipo indefinido. Iremos analizando cada una de ellas de manera individual, y en caso de considerar conveniente, realizaremos la comparación con nuestra variable a predecir.
# 
# Vamos a observar un resumen sobre las variables numéricas:

# In[4]:


data.describe()


# Mostramos la matriz de correlación entre las variables

# In[5]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# Juntamos las columnas de año, mes y dia para crear un único datetime.
# También añadimos una con el mes de forma numérica para usarlas posteriormente.

# In[6]:


data['arrival_datetime'] = pd.to_datetime(data['arrival_date_year'].map(str) + "-" + data['arrival_date_month'].map(str) + "-" + data['arrival_date_day_of_month'].map(str))
data['arrival_month'] = data['arrival_datetime'].map(str).str[5:7]


# In[7]:


data = data.sort_values(by=['arrival_datetime'], ascending=True)
data


# Agrupamos los datos por la fecha total, y por año y mes para estudiar los cambios a lo largo del tiempo.

# In[8]:


data_groupby_year_month = data.groupby(by=["arrival_date_year", "arrival_date_month"])
data_groupby_date = data.groupby(by=["arrival_datetime"])


# ### hotel
# Primero comenzamos observando la columna de hotel
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


# Creamos una tabla de contiengencias, y después esa misma tabla de contingencias pero observando los porcentajes, de tal forma que se puede ver que el porcentaje de cancelados es mayor en "City Hotel".
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
# Pintar todos los valores del eje X!!!


# ### is_canceled
# 
# Se trata de unas de las variables más importantes, pues sería interesante tratar de predecirla en un futuro.

# In[14]:


data['is_canceled'].describe()


# In[15]:


data['is_canceled'].plot(kind='bar')


# ##### Evolución histórica

# In[16]:


data_canceled_and_date = data[['arrival_date_year', 'arrival_month', 'is_canceled']].groupby(by=['arrival_date_year', 'arrival_month']).sum()
data_canceled_and_date.head()


# In[17]:


plt.figure(figsize=(500,10));
data_canceled_and_date.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### lead_time 	
# 

# In[18]:


data['lead_time'].describe()


# In[19]:


data['lead_time'].hist()


# ##### Evolución histórica
# Vamos a mostrar como ha ido evolucionando la variable a lo largo de cada mes.

# In[20]:


data_lead_time_and_date = data[['arrival_date_year', 'arrival_month', 'lead_time']].groupby(by=['arrival_date_year', 'arrival_month']).sum()
data_lead_time_and_date.head()


# In[21]:


plt.figure(figsize=(500,10));
data_lead_time_and_date.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# In[22]:


data_lead_time_and_date.plot.scatter(x=['arrival_date_year', 'arrival_month'], y=['lead_time'])


# ### arrival_date_year 	

# In[ ]:


data['arrival_date_year'].describe()


# In[ ]:


plt.xticks(rotation=45)
data['arrival_date_year'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del año y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_year = pd.crosstab(data['arrival_date_year'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_year


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_year.plot.bar(figsize=(30,8));
plt.tight_layout()


# ##### Comparación con is_canceled
# Vamos a agrupar por año para ver como han ido cambiando las cancelaciones a lo largo de los años.

# In[ ]:


data_year_and_canceled = data[['arrival_date_year', 'is_canceled']]
data_year_and_canceled = data_year_and_canceled.groupby(by=['arrival_date_year']).sum()
data_year_and_canceled


# In[ ]:


plt.figure(figsize=(500,10));
data_year_and_canceled.plot(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### arrival_month 	
# Usamos arrival_month(numérica), en lugar de arrival_date_month(categórica), pues son equivalentes.

# In[ ]:


data['arrival_month'].describe()


# In[ ]:


data['arrival_month'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del mes y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_month = pd.crosstab(data['arrival_month'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_month


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_month.plot(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# #### Comparación con is_canceled
# Vamos a agrupar por mes para ver como han ido cambiando las cancelaciones a lo largo de los meses.

# In[ ]:


data_month_and_canceled = data[['arrival_month', 'is_canceled']]
data_month_and_canceled = data_month_and_canceled.groupby(by=['arrival_month']).sum()
data_month_and_canceled


# In[ ]:


plt.figure(figsize=(500,10));
data_month_and_canceled.plot(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### arrival_date_week_number 	

# In[ ]:


data['arrival_date_week_number'].describe()


# In[ ]:


data['arrival_date_week_number'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función de la semana y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_weeknumber = pd.crosstab(data['arrival_date_week_number'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_weeknumber.head()


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_weeknumber.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# #### Comparación con is_canceled
# Vamos a agrupar por el número de la semana para ver como han ido cambiando las cancelaciones a lo largo de las mismas.

# In[ ]:


data_week_and_canceled = data[['arrival_date_week_number', 'is_canceled']]
data_week_and_canceled = data_week_and_canceled.groupby(by=['arrival_date_week_number']).sum()
data_week_and_canceled.head()


# In[ ]:


plt.figure(figsize=(500,10));
data_week_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### arrival_date_day_of_month 	

# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del día del mes y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_daynumber = pd.crosstab(data['arrival_date_day_of_month'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_daynumber.head()


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_daynumber.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ##### Comparación con is_canceled
# Vamos a agrupar por día del mes para ver como han ido cambiando las cancelaciones a lo largo de los mismos.

# In[ ]:


data_day_and_canceled = data[['arrival_date_day_of_month', 'is_canceled']]
data_day_and_canceled = data_day_and_canceled.groupby(by=['arrival_date_day_of_month']).sum()
data_day_and_canceled.head()


# In[ ]:


plt.figure(figsize=(500,10));
data_day_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### stays_in_weekend_nights 	

# In[ ]:


data['stays_in_weekend_nights'].describe()


# In[ ]:


data['stays_in_weekend_nights'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de noches de fin de semana y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_weekend_nights = pd.crosstab(data['stays_in_weekend_nights'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_weekend_nights


# Aquí se pueden ver algunos detalles significativos. Cuando se reservan 10 noches de fin de semana NADIE ha cancelado nunca la reserva. Y cuando se reservan 7 noches de fin de semana, TODO el mundo ha cancelado la reserva.

# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_weekend_nights.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ##### Comparación con is_canceled
# Vamos a agrupar por la variable para ver como han ido cambiando las cancelaciones en función de la misma.

# In[ ]:


data_weekend_nights_and_canceled = data[['stays_in_weekend_nights', 'is_canceled']]
data_weekend_nights_and_canceled = data_weekend_nights_and_canceled.groupby(by=['stays_in_weekend_nights']).sum()
data_weekend_nights_and_canceled


# Al ver estos resultado se aprecia que la probabilidad de antes es engañosa, ya que se debe a muestras muy pequeñas o incluso vacías.

# In[ ]:


plt.figure(figsize=(500,10));
data_weekend_nights_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### stays_in_week_nights 	

# In[ ]:


data['stays_in_week_nights'].describe()


# In[ ]:


data['stays_in_week_nights'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de noches entre semana y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_week_nights = pd.crosstab(data['stays_in_week_nights'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_week_nights


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_week_nights.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ##### Comparación con is_canceled
# Vamos a agrupar por la variable para ver como han ido cambiando las cancelaciones a lo largo de la misma.

# In[ ]:


data_week_nights_and_canceled = data[['stays_in_week_nights', 'is_canceled']]
data_week_nights_and_canceled = data_week_nights_and_canceled.groupby(by=['stays_in_week_nights']).sum()
data_week_nights_and_canceled


# In[ ]:


plt.figure(figsize=(500,10));
data_week_nights_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### adults 	

# In[ ]:


data['adults'].describe()


# In[ ]:


data['adults'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de adultos y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_adults = pd.crosstab(data['adults'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_adults


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_adults.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# #### Comparación con is_canceled
# Vamos a agrupar por el número de adultos para ver como han ido cambiando las cancelaciones en función de los mismos.

# In[ ]:


data_adults_and_canceled = data[['adults', 'is_canceled']]
data_adults_and_canceled = data_adults_and_canceled.groupby(by=['adults']).sum()
data_adults_and_canceled


# In[ ]:


plt.figure(figsize=(500,10));
data_adults_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### children 	

# In[ ]:


data['children'].describe()


# In[ ]:


data['children'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del número de niños y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_children = pd.crosstab(data['children'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_children


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_children.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# #### Comparación con is_canceled
# Vamos a agrupar por el número de niños para ver como han ido cambiando las cancelaciones en función de los mismos.

# In[ ]:


data_children_and_canceled = data[['children', 'is_canceled']]
data_children_and_canceled = data_children_and_canceled.groupby(by=['children']).sum()
data_children_and_canceled


# In[ ]:


plt.figure(figsize=(500,10));
data_children_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### babies

# In[ ]:


data['babies'].describe()


# In[ ]:


data['babies'].hist()


# ##### Tabla de contingencias
# Sacamos una tabla de contingencias para ver con que probabilidad se cancela o no en función del nñumero de bebés y pintarlo en una gráfica.

# In[ ]:


data_prob_canceled_by_babies = pd.crosstab(data['babies'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_babies


# In[ ]:


plt.figure(figsize=(500,10));
data_prob_canceled_by_babies.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# #### Comparación con is_canceled
# Vamos a agrupar por el número de bebés para ver como han ido cambiando las cancelaciones en función de los mismos.

# In[ ]:


data_babies_and_canceled = data[['babies', 'is_canceled']]
data_babies_and_canceled = data_babies_and_canceled.groupby(by=['babies']).sum()
data_babies_and_canceled


# In[ ]:


plt.figure(figsize=(500,10));
data_babies_and_canceled.plot.bar(figsize=(30,8));
plt.tight_layout()
# Pintar todos los valores del eje X!!!


# ### meal
# 

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

# In[ ]:


data.meal.describe()


# In[ ]:


data.meal.value_counts()


# In[ ]:


data.meal.hist()


# In[ ]:


ax = sns.stripplot(y="stays_in_weekend_nights", x="meal",hue="is_canceled", data=data)


# ### country

# La variable country representa el país de la reserva, d emodo que constituye una variable categórica con 108 valores diferentes. El pais mas representado es Portugal.

# In[ ]:


data.country.describe()


# In[ ]:


data.country.value_counts()


# Vemos que el pais del cual mas información poseemos es portugal, seguido de Reino Unido y Francia. Analizaremos el numero de cancelaciones por país de los 5 más abundantes

# In[ ]:


data_topcountrys= data[data.country=='PRT']
data_topcountrys= data_topcountrys.append(data[data.country=='GBR'])
data_topcountrys= data_topcountrys.append(data[data.country=='FRA'])
data_topcountrys= data_topcountrys.append(data[data.country=='ESP'])
data_topcountrys= data_topcountrys.append(data[data.country=='DEU'])
data_prob_canceled_by_country = pd.crosstab(data_topcountrys['country'], data_topcountrys['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_country


# In[ ]:


data_prob_canceled_by_country.plot.bar(figsize=(30,8))


# Podemos observar que el ratio de cancelaciones en portugal es bastante mas alto al del resto de los paises top de nuestro dataset.

# ### market_segment

# La variable market segment representa el segmento de mercado que ha realizado la contratacion. Es una varibale categorica con 7 valores distintos.
# El término "TA" significa "Agentes de Viaje" y "TO" significa "Operadores Turísticos"

# In[ ]:


data.market_segment.describe()


# In[ ]:


data.market_segment.value_counts()


# In[ ]:


data_prob_canceled_by_market = pd.crosstab(data['market_segment'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_market


# In[ ]:


data_prob_canceled_by_market.plot.bar(figsize=(30,8))


# Podemos observar que las reservas de grupos son las mas propensas a cancelar las reservas, siendo esta la opcion mas habital. En el lado opuesto tenemos el caso de la aviación.

# ### distribution_channel

# Canal de distribución de reservas. El término "TA" significa "Agentes de Viaje" y "TO" significa "Operadores Turísticos"

# In[ ]:


data.distribution_channel.describe()


# In[ ]:


data.distribution_channel.value_counts()


# In[ ]:


data_prob_canceled_by_market = pd.crosstab(data['distribution_channel'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled_by_market


# In[ ]:


data_prob_canceled_by_market.plot.bar(figsize=(30,8))


# In[ ]:


Podemos observar que las reservas por TA/TO suelen cancelarse más amenudo


# ### is_repeated_guest

# Valor que indica si el nombre de la reserva era de un huésped repetido (1) o no (0)

# In[ ]:


data.is_repeated_guest.describe()


# In[ ]:


data.is_repeated_guest.sum()


# In[ ]:


data_repeted_guest=data[data.is_repeated_guest==1]
data_repeted_guest.is_canceled.sum()


# In[ ]:


data_repeted_guest=data[data.is_repeated_guest==0]
data_repeted_guest.is_canceled.sum()


# Podemos observar que de los repetidor, solo 52 cancelaron su viaje, es decir, 16%. Frente a los no repetidores, entre los que la tasa de cancelacion es 38%

# ### previous_cancellations

# Número de reservas anteriores que fueron canceladas por el cliente antes de la reserva actual

# In[ ]:


data.previous_cancellations.describe()


# In[ ]:


data.previous_cancellations.value_counts()


# In[ ]:


data_prob_canceled = pd.crosstab(data['previous_cancellations'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_canceled


# In[ ]:


data_prob_canceled.plot.bar(figsize=(30,8))


# Podemos observar, que la tonica general es no tener cancelaciones previas. Pero a mayor numero de cancelaciones previas, se cancela menos la reserva.

# ### previous_bookings_not_canceled

# Número de reservas anteriores no canceladas por el cliente antes de la reserva actual

# In[ ]:


data.previous_bookings_not_canceled.describe()


# In[ ]:


data.previous_bookings_not_canceled.value_counts()


# In[ ]:


data_prob_ncanceled = pd.crosstab(data['previous_bookings_not_canceled'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_ncanceled


# In[ ]:


data_prob_ncanceled.plot.bar(figsize=(30,8))


# Podemos observar, que el comportamiento es similar al caso anterior, es decir, aquellos con pocas cancelaciones previas, es mas probable que cancelen, pero a mayor numero de cancelaciones general, es mas dificil que sea cancelado.

# ### reserved_room_type

# Código de tipo de habitación reservada. El código se presenta en lugar de la designación por razones de anonimato.

# In[ ]:


data.reserved_room_type.describe()


# In[ ]:


data.reserved_room_type.value_counts()


# In[ ]:


data_prob_res = pd.crosstab(data['reserved_room_type'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[ ]:


data_prob_res.plot.bar(figsize=(30,8))


# Podemos observar que las cancelaciones son similares en todos los tipos de habitacion reservada, salvo en 3 casos llamativos:
# * Tipo H donde las cancelaciones son ligeramente superiores
# * Tipo L no existen cancelaciones
# * Tipo P solo existen cancelaciones

# ### assigned_room_type

# Código del tipo de habitación asignada a la reserva. A veces el tipo de habitación asignada difiere del tipo de habitación reservada debido a razones de operación del hotel (por ejemplo, sobreventa) o por petición del cliente. El código se presenta en lugar de la designación por razones de anonimato.

# In[ ]:


data.assigned_room_type.describe()


# In[ ]:


data.assigned_room_type.value_counts()


# Puede ser interesante si los tipos assignados son los tipos reservados, y la relación que ello tiene con la cancelación de la reserva

# In[ ]:





# In[ ]:





# ### booking_changes

# Número de cambios/enmiendas hechos a la reserva desde el momento en que la reserva fue ingresada en el PMS hasta el momento del registro o cancelación.

# In[ ]:


data.booking_changes.describe()


# In[ ]:


data.booking_changes.value_counts()


# In[ ]:


data_prob_res = pd.crosstab(data['booking_changes'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[ ]:


data_prob_res.plot.bar(figsize=(30,8))


# Podemos destacar que las cancelaciones aumentan cuando sufren un mayor numero de cambios en las reservas, ademas para cambios pequeños, las reservas no se resienten en terminos de cancelación

# ### deposit_type

# Indicación sobre si el cliente hizo un depósito para garantizar la reserva. Esta variable puede asumir tres categorías: No Deposito - no se hizo ningún depósito; No Reembolso - se hizo un depósito con un valor por debajo del costo total de la estancia.

# In[ ]:


data.deposit_type.describe()


# In[ ]:


data.deposit_type.value_counts()


# In[ ]:


data_prob_res = pd.crosstab(data['deposit_type'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# In[ ]:


data_prob_res.plot.bar(figsize=(30,8))


# Muy destacable que las reservas sin refound son las mas canceladas

# ### agent
# 

# La identificación de la agencia de viajes que hizo la reserva

# In[ ]:


data.agent.describe()


# In[ ]:


data.agent.value_counts()


# In[ ]:


data_prob_res = pd.crosstab(data['agent'], data['is_canceled']).apply(lambda r: r/r.sum(), axis=1)
data_prob_res


# Mucha cardinalidad en la variable, dado que hay 500+ valores de agencia y no tenemos traduccion de estos id (VAR CANDIDATA A DESAPARECER)

# In[ ]:





# In[ ]:




