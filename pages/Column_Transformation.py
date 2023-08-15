import streamlit as st
import pandas as pd
import src.csv_llm as ut
import json

@st.cache_data
def get_trans_func(template_df,target_df,col_map):
    return ut.get_trans_func(template_df,target_df,col_map)

st.title("Column Transformation Function")
col1, col2,col3,col4 = st.columns(4)

func_res = get_trans_func(st.session_state.TEMPLATE_DF,st.session_state.TARGET_DF,st.session_state.FINAL_COL_MAP)

if "FUNC_MAP" not in st.session_state:
    st.session_state.FUNC_MAP = func_res
for temp_col,tar_col in st.session_state.FINAL_COL_MAP.items():
    with col1:
        st.write("Template Column")
        st.dataframe(st.session_state.TEMPLATE_DF.dropna().head(4)[temp_col],hide_index=True)
        
    with col2:
        st.write("Transformed Column")
        try:
            tar_col = st.session_state.TARGET_DF.dropna().head(4)[tar_col]
            tar_col = tar_col.apply(ut.string_to_function(st.session_state.FUNC_MAP[temp_col]["func_str"]))
            st.dataframe(tar_col,hide_index=True)
        except Exception as e:
            st.error(str(e))
            
    with col3:
        st.session_state.FUNC_MAP[temp_col]["func_str"] = st.text_area("Edit Code", value=st.session_state.FUNC_MAP[temp_col]["func_str"], height=200,key=f"{temp_col}_{tar_col}")   

    with col4:
        st.write("Transform code")
        st.code(st.session_state.FUNC_MAP[temp_col]["func_str"], language="python")

st.markdown(f"##### CLICK ON `Final Data` TAB ON LEFT SIDE PANE TO DOWNLOAD TRANSFORMED CSV")