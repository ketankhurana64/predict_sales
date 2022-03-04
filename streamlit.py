import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import time


allcolumns=['shop_id','item_id','day','month','year']

shopid = st.text_input('Input shop id: ')
itemid = st.text_input('Input item id: ')
if shopid and itemid:
    try:
        shopid=int(shopid)
        itemid=int(itemid)
    except ValueError:
        st.write('Please input integer values only.')        

    else:
        day='01'
        month='01'
        year='2016'
        df_test=pd.DataFrame(data=[[shopid,itemid,day,month,year]],columns=allcolumns)

        model=pickle.load(open('model_decisiontree.pkl','rb'))        
    
        import joblib
        min_max_scaler=joblib.load('scaler.save')
        df_test_scaled = min_max_scaler.transform(df_test)        
    
        output=model.predict(df_test_scaled)
        st.write("Shop id "+str(shopid)+ " is expected to sell "+str(round(output[-1]))+ " item(s) next month")
