import streamlit as st
import pandas as pd
import src.csv_llm as ut
import json
import base64


tar_df = st.session_state.TARGET_DF.copy()
tar_cols = [val["col_name"] for val in  st.session_state.FUNC_MAP.values()]
tar_df = tar_df[tar_cols]
for temp_col,val in st.session_state.FUNC_MAP.items():
    tar_df[val["col_name"]] = tar_df[val["col_name"]].apply(ut.string_to_function(st.session_state.FUNC_MAP[temp_col]["func_str"]))

rename_map = {val["col_name"]:temp_col for temp_col,val in  st.session_state.FUNC_MAP.items()}
tar_df = tar_df.rename(columns=rename_map)

st.write("Sample Template Data")
st.dataframe(st.session_state.TEMPLATE_DF.head(5),hide_index=True)

st.write("Sample Target Data")
st.dataframe(tar_df.head(5),hide_index=True)


csv = tar_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download Transformed CSV file</a>'
st.markdown(href, unsafe_allow_html=True)

st.markdown(f"##### CLICK ON `Upload CSV` TAB ON LEFT SIDE PANE TO PROCESS CSV.")

    