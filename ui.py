import streamlit as st 
import pandas as pd
import joblib 
import os 
import numpy as np
from orchestrator import AuraML

st.set_page_config(page_title='AuraML Dashboard', layout="wide")
st.title("AuraML Intelligence Dashboard")

@st.cache_resource
def load_engine():
    if os.path.exists('auraml_engine.joblib'):
        return joblib.load('auraml_engine.joblib')
    return None

engine = load_engine()

if engine is None:
    st.error("❌ 'auraml_engine.joblib' not found. Run your training script first!")
else:
    st.sidebar.header('Engine Metadata')
    st.sidebar.info(f"**Task:** {engine.custodian.task_type.upper()}")
    st.sidebar.info(f"**Target:** {engine.target_col}")

    uploaded_file = st.file_uploader("Upload CSV for prediction", type='csv')

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # We create a placeholder for the results so the UI feels responsive
        result_placeholder = st.empty()
        
        # Show initial preview
        st.write('### Input Data Preview')
        st.dataframe(df.head(10))

        if st.button("Run High-Performance Inference"):
            with st.spinner("AuraML Engine is calculating..."):
                try:
                    # 1. Execute Prediction
                    # Since you didn't change the handler, we assume 
                    # these are standard numerical outputs.
                    preds = engine.predict(df)
                    
                    # 2. Assign to a COPY of the dataframe to ensure clean state
                    processed_df = df.copy()
                    processed_df['AuraML_Prediction'] = preds
                    
                    # 3. Use columns to show the success and the specific results
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write("### Predictions Top 10")
                        st.dataframe(processed_df[['AuraML_Prediction']].head(10))
                    
                    with col2:
                        # Quick summary stats to verify it's working
                        st.write("### Prediction Stats")
                        st.write(f"**Average Value:** {np.mean(preds):.2f}")
                        st.write(f"**Max Value:** {np.max(preds):.2f}")

                    # 4. Display the FULL table with the results
                    st.write("### Full Results Table")
                    st.dataframe(processed_df)

                    # 5. Provide Download Button
                    csv = processed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='auraml_predictions.csv',
                        mime='text/csv'
                    )

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.info("Ensure your CSV has the exact same columns used during training.")