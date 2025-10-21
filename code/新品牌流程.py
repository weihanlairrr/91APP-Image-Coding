import streamlit as st
import os
import time
from streamlit_extras.stylable_container import stylable_container

def tab4():
    st.write("\n")
    def _mapping_upload(brand: str, uploaded_file):
        """
        Callback：建立資料夾、複製並重新命名對照表，
        然後遞增 mapping_idx 並記錄最後儲存路徑。
        """
        dep_base = r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies"
        target_dir = os.path.join(dep_base, brand)
        os.makedirs(target_dir, exist_ok=True)
    
        target_filename = f"{brand}_檔名角度對照表.xlsx"
        target_path = os.path.join(target_dir, target_filename)
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
        # 更新索引，使下一次 widget 變成空白
        st.session_state["mapping_idx"] += 1
        # 記錄儲存路徑，用於後面顯示成功訊息
        st.session_state["mapping_last_path"] = target_path
    
    # ======================================================================
    # 1. 建立檔名角度對照表（包在淺灰底 container 裡）
    # ======================================================================
    with stylable_container(
        key="mapping_section_container1",
        css_styles="""
        {
          background-color: #e8e8e8;
          padding-top: 1rem;
          padding-left: 2rem;
          padding-right: 1rem;
          padding-bottom: 2rem;
          border-radius: 6px;
        }
        """
    ):
        st.subheader("1. 建立檔名角度對照表 + 上傳")
    
        # 1. 初始化 mapping_idx（只做一次）
        if "mapping_idx" not in st.session_state:
            st.session_state["mapping_idx"] = 0
    
        step = st.pills("step", ["Step1", "Step2"], label_visibility="collapsed",default="Step1",key= "pills1")
    
        if step == "Step1":
            st.write("\n")
            col1, col2, col3, col4 = st.columns([0.5, 0.2, 1.4, 2.7], vertical_alignment="center")
            template_path = r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies\檔名角度對照表_範本.xlsx"
            template_filename = "檔名角度對照表_範本.xlsx"
            if os.path.exists(template_path):
                with open(template_path, "rb") as file:
                    col1.download_button(
                        label="下載範本",
                        data=file,
                        file_name=template_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error(f"錯誤：找不到範本檔案，請確認路徑是否正確。\n預期路徑: {template_path}")
            col2.markdown('➔', unsafe_allow_html=True)
            col3.markdown(
                '參考<a href="https://docs.google.com/presentation/d/1KCjiFoVW4o2q4Dhok0v1Kd7Hp-fJt_t7K7feRx22zSY/edit?usp=sharing" target="_blank">簡報</a>(第7頁開始)填寫表格',
                unsafe_allow_html=True
            )
    
        elif step == "Step2":
            colB, colC, colD = st.columns([6.8, 0.7, 2.5], vertical_alignment="center")
            # 2. 動態 key：每次上傳成功後 mapping_idx +1，下一次就自動清空
            idx       = st.session_state["mapping_idx"]
            brand_key = f"mapping_brand_input_{idx}"
            file_key  = f"mapping_file_uploader_{idx}"
            btn_key   = f"mapping_upload_btn_{idx}"
    
            # 2.1 輸入品牌簡稱
            with colB:
                cola, colb = st.columns([2, 4.8], vertical_alignment="top")
                brand_input = cola.text_area(
                    "請輸入品牌簡稱",
                    key=brand_key,
                    placeholder="輸入品牌簡稱",
                    label_visibility="collapsed",
                    height=68
                )
                # 2.2 上傳 .xlsx
                with colb:
                    with stylable_container(
                        key="file_uploader_newbrand",
                        css_styles="""
                        {
                          [data-testid='stFileUploaderDropzoneInstructions'] > div > span {
                            display: none;
                          }
                          [data-testid='stFileUploaderDropzoneInstructions'] > div::before {
                            content: '請上傳檔名角度對照表';
                          }
                        }
                        """
                    ):
                        uploaded_mapping = st.file_uploader(
                            "上傳檔名角度對照表 (.xlsx)",
                            type=["xlsx"],
                            key=file_key,
                            label_visibility="collapsed"
                        )
    
            # 2.3 當兩者都有值時，顯示「上傳」按鈕
            if brand_input and uploaded_mapping:
                if colD.button(
                    "上傳",
                    key=btn_key,
                    on_click=_mapping_upload,
                    args=(brand_input, uploaded_mapping)
                ):
                    # 按鈕被點擊後，_mapping_upload 會先執行
                    pass
    
            # 3. 獨立顯示成功訊息，置於邏輯最後面
            if "mapping_last_path" in st.session_state:
                placeholder = colD.empty()
                placeholder.success("上傳完成")
                # 等候 2 秒後清除訊息
                time.sleep(2)
                placeholder.empty()
                # 清除狀態，避免重複顯示
                del st.session_state["mapping_last_path"]


    st.write("\n")
    with stylable_container(
        key="mapping_section_container2",
        css_styles="""
        {
          background-color: #e8e8e8;
          padding-top: 1rem;
          padding-left: 2rem;
          padding-right: 1rem;
          padding-bottom: 2rem;
          border-radius: 6px;
        }
        """
    ):
        st.subheader("2. 製作圖片樣本集 + 訓練")
        stepb = st.pills("step", ["Step1", "Step2"], label_visibility="collapsed",default="Step1", key= "pills2")
        if stepb == "Step1":
            st.write("\n")
            template_path = r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\資料集\圖片樣本集_範本.xlsx"
            template_filename = "圖片樣本集_範本.xlsx"
            if os.path.exists(template_path):
                with open(template_path, "rb") as file:
                    st.download_button(
                        label="下載樣本集範本",
                        data=file,
                        file_name=template_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            with st.expander("",expanded=True):
                col1, col2, col3 =st.columns([2,2,1])
                img_path1_1 =  r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies\樣本集示範2.png"
                col1.image(img_path1_1, caption="根據檔名角度對照表統整各分類的角度，並寫出優先順序")
                
                img_path1_2 =  r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies\樣本集示範3.png"
                col2.image(img_path1_2, caption="如有自訂的虛構角度也記得放入對應順序(此範例中為HERO)")
                
                img_path2 = r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies\樣本集示範1.png"
                st.image(img_path2, caption="建議使用圖床網址。編號欄為公式")
        elif stepb == "Step2":
            st.write("\n")
            st.write("資料庫優化  ➔  上傳樣本集  ➔  執行  ➔  耐心等待跑玩完")
            img_path4 = r"G:\共用雲端硬碟\TP代營運\1-小工具\自動編圖工具\dependencies\資料庫優化示範.png"
            col1,col2 =st.columns([3,1.5])
            col1.image(img_path4, caption="進度條還沒跑完前請勿關閉螢幕。可切換視窗做其他事。")
            