import streamlit as st 
import pandas as pd
import joblib 
import os 

st.set_page_config(page_title = 'AuraML Dashboard')
st.title ("AuraML Intelligence Dashboard")
st.markdown("Upload your data and let the Engine handle the rest")

if not os.path.exists('auraml_engine.joblib'):
    st.warning("No AuraML_Engine.joblib found, please run the trianing scripts first to generate your joblib file")

else :
    engine= joblib.load('auraml_engine.joblib')
    
    st.sidebar.header('Model Info')
    st.sidebar.write(f"**Task : ** {engine.custodian.task_type.upper()}")
    st.sidebar.write(f"**Target col :** {engine.target_col}")


    uploaded_file= st.file_uploader(" Upload csv for prediction", type='csv')

    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        st.write('### Data Preview', df.head())

        if st.button("Run Inference"):
            with st.spinner("Processing ..."):
                try:
                    preds= engine.predict(df)
                    df['AuraML Predictions']= preds
                    st.success("Predictions complete")
                    st.dataframe(df)

                    csv= df.to_csv(index =False).encode('utf-8')
                    st.download_button('Downloaded results', csv , 'predictions.csv','text/csv')

                except Exception as e:
                    st.error(f"error: {e}")
            