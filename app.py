import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "code"))

import streamlit as st
import warnings
import logging
import faiss  
import 自動編圖
import 編圖複檢
import 圖片訓練
import 批次換圖
from style import custom_css

st.set_page_config(page_title='TP自動編圖工具', page_icon='👕', layout="wide")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
faiss.omp_set_num_threads(1)
st.markdown(custom_css, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["自動編圖", "編圖複檢", "圖片訓練","ADS批次換圖"])

with tab1:
    自動編圖.tab1()

with tab2:
    編圖複檢.tab2()

with tab3:
    圖片訓練.tab3()
 
with tab4:
    批次換圖.tab4()
