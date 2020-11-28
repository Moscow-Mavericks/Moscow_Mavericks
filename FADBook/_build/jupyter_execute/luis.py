#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime


# In[2]:


datos_booking=pd.read_csv("EDAbooking.csv",delimiter=";")
datos_booking.head()
df_booking=pd.DataFrame(datos_booking)
df_booking.columns


# In[3]:


df_EDA_variables=pd.DataFrame(df_booking, columns=['deposit_type', 'agent',
       'company', 'days_in_waiting_list', 'customer_type', 'adr',
       'required_car_parking_spaces', 'total_of_special_requests',
       'reservation_status', 'reservation_status_date'])
df_EDA_variables


# In[13]:


df_EDA_variables.info()


# ### deposit_type: 
# Variable categórica (cualitativa) que describe si los clientes han efectuado depósitos anticipados.

# In[30]:


print(df_EDA_variables.deposit_type.unique())
cuenta_No_deposit=0
cuenta_Non_refund=0
cuenta_Refundable=0
for elemento in df_EDA_variables.deposit_type:
    if (elemento == "No Deposit"):
        cuenta_No_deposit=cuenta_No_deposit + 1
    if (elemento == "Non Refund"):
        cuenta_Non_refund=cuenta_Non_refund + 1
    if (elemento == "Refundable"):
        cuenta_Refundable = cuenta_Refundable + 1

frecuencia_No_deposit= cuenta_No_deposit/100
print("El porcentaje de reservas sin depositos es: ", frecuencia_No_deposit,"%")

frecuencia_Non_refund= cuenta_Non_refund/100
print("El porcentaje de depositos sin reembolso es: ",frecuencia_Non_refund,"%")

frecuencia_Refundable=cuenta_Refundable/100
print("El porcentaje de despositos reembolsables es: ",frecuencia_Refundable,"%")

df_EDA_variables["deposit_type"].describe()


# In[16]:


df_EDA_variables["deposit_type"].hist()


# In[32]:


## Comparación respecto is_canceled
pd.crosstab(datos_booking['deposit_type'], datos_booking['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# ### Agent
# la variable AGENT (cualitativa) esun id que indica la agencia de viajes que hace la reserva.

# In[29]:


id_agencias=df_EDA_variables.agent.unique()
frecuencia_id_agencias=df_EDA_variables.agent.value_counts()
porcentaje_frecuencias_id_agencias=df_EDA_variables.agent.value_counts(9.0)
print("El número de reserva hechas por cada id:",frecuencia_id_agencias)

print("El porcentaje de reservas por cada id:", porcentaje_frecuencias_id_agencias)


# ### Company
# La variable company es un identificador de la empresa o entidad que realiza la reserva.

# In[19]:


id_compañias=df_EDA_variables.company.unique()

frecuencia_id_compañia=df_EDA_variables.company.value_counts()
porcentaje_frecuencias_id_compañia=df_EDA_variables.company.value_counts(40.0)
print("El número de reserva hechas por cada id:",frecuencia_id_compañia)
print("El porcentaje de reservas por cada id:", porcentaje_frecuencias_id_compañia)


# ### Days_in_waiting_list
# Variable cuantitativa que recoge el número de días que pasan hasta que el usuario confirma la reserva.

# In[9]:


df_EDA_variables["days_in_waiting_list"].describe()


# In[10]:


df_EDA_variables["days_in_waiting_list"].hist()


# In[11]:


## comparación respecto a is_canceled
pd.crosstab(datos_booking['days_in_waiting_list'], datos_booking['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# ### Customer_type:
# variable categórica que clasifica los tipos d ereservas en:contractuales, grupales, transitorias o perteneciente a una transitoria.

# In[12]:


df_EDA_variables["customer_type"].describe()


# In[13]:


df_EDA_variables["customer_type"].hist()


# In[14]:


## Comparación respecto is_canceled.
pd.crosstab(datos_booking['customer_type'], datos_booking['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# ## ADR:
# Variable cuantitativa que refleja el ratio entre precio pagado por la reserva o estancia, dividido entre el número de noches de estancia del usuario.

# In[23]:


df_EDA_variables["adr"].describe()


# In[24]:


df_EDA_variables["adr"].hist()


# In[25]:


pd.crosstab(datos_booking['adr'], datos_booking['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# ### Required_car_parking_spaces: 
# Número de plazas de parking solicitadas por el usuario.

# In[15]:


df_EDA_variables["required_car_parking_spaces"].describe()


# In[16]:


df_EDA_variables["required_car_parking_spaces"].hist()


# In[17]:


## Comparación respecto is_canceled.
pd.crosstab(datos_booking['required_car_parking_spaces'], datos_booking['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# ### Total_of_special_requests: 
# número total de requisitos especiales solicitados por el usuario.

# In[18]:


df_EDA_variables["total_of_special_requests"].describe()


# In[19]:


df_EDA_variables["total_of_special_requests"].hist()


# In[ ]:


## Comparación respecto is_canceled
pd.crosstab(datos_booking['total_of_special_requests'], datos_booking['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# ### reservation_status: 
# Último estado de la reserva, variable categórica que clasifica en: cancelado, check-out (finalizada estancia), sin aparecer (el usuario no ha realizado el check in, informando al hotel la razón)
# 

# In[21]:


df_EDA_variables["reservation_status"].describe()


# In[22]:


df_EDA_variables["reservation_status"].hist()


# In[23]:


## Comparación respecto is_canceled
pd.crosstab(datos_booking['reservation_status'], datos_booking['is_canceled']).apply(lambda r: r/r.sum(), axis=1)


# ### reservation_status_date
# Fecha de la última actualización del estado de la reserva

# In[4]:


df_EDA_variables["reservation_status_date"].describe()


# In[37]:


fechas=pd.to_datetime(df_EDA_variables["reservation_status_date"])
fechas.hist()

