import streamlit as st
import pandas as pd
import src.csv_llm as ut
import json

FINAL_COL_MAP = {}
TEMPLATE_DF = None
TARGET_DF = None

@st.cache_data
def get_col_map(target_df,template_df):
    return ut.get_col_map(target_df,template_df)

st.title("Column Selection")
# Upload CSV Files
st.sidebar.header("Upload Files")
template_file = st.sidebar.file_uploader("Upload Template CSV File", type=["csv"], key="template")
target_file = st.sidebar.file_uploader("Upload Target CSV File", type=["csv"], key="target")

if template_file and target_file:
    TEMPLATE_DF = pd.read_csv(template_file)
    TARGET_DF = pd.read_csv(target_file)
    
   
    col_map = get_col_map(TARGET_DF,TEMPLATE_DF)
  
    temp_cols = TEMPLATE_DF.columns.tolist()
    
    for i, temp_col in enumerate(temp_cols):
        st.markdown(f"**Template column: {temp_col}**")
        tar_col = col_map[temp_col]['col_name']
        cand_cols = [val.strip() for val in col_map[temp_col]['similar_col_name'].split(",")]
        other_cols = list(set(TARGET_DF.columns.tolist()) - set(cand_cols+[tar_col]))
        
        col1, col2,col3,col4,col5 = st.columns(5)
        with col1:
            radio_choice = st.radio("Select option", ("Predicted Similar Column", "Other Similar Columns", "Other Columns"),key=f"Radio_{i}")

        with col2:
            if radio_choice=="Predicted Similar Column":
                st.write("Predicted Similar Column")
                st.markdown(f"**{tar_col}**")
                FINAL_COL_MAP[temp_col]=tar_col
            elif radio_choice=="Other Similar Columns":
                FINAL_COL_MAP[temp_col]=st.selectbox("Select other similar columns",cand_cols,key=f"Dropdown_{i}")
            else:
                FINAL_COL_MAP[temp_col]= st.selectbox("Select other columns", other_cols,key=f"Dropdown_{i}")
        
        with col3:
            st.write("Template Column")
            st.dataframe(TEMPLATE_DF.dropna().head(3)[temp_col],hide_index=True)
        with col4:
            st.write("Target Column")
            st.dataframe(TARGET_DF.dropna().head(3)[FINAL_COL_MAP[temp_col]],hide_index=True)
        
        with col5:
            st.write("Explanation")
            st.write(col_map[temp_col]['reason'])
            

        st.session_state["TEMPLATE_DF"]=TEMPLATE_DF
        st.session_state["TARGET_DF"]=TARGET_DF
        st.session_state["FINAL_COL_MAP"]=FINAL_COL_MAP
        
    st.markdown(f"##### CLICK ON `Column Transformation` TAB ON LEFT SIDE PANE TO GET COLUMN TRANSFORMATION FUNCTION")
        