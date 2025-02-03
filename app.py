import streamlit as st
import os
import faiss  
import dependencies.自動編圖
import dependencies.編圖複檢
from dependencies.style import custom_css

st.set_page_config(page_title='TP自動編圖工具', page_icon='👕', layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
faiss.omp_set_num_threads(1)
st.markdown(custom_css, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["自動編圖", "編圖複檢"])

with tab1:
    dependencies.自動編圖.tab1()

with tab2:
    dependencies.編圖複檢.tab2()
 
