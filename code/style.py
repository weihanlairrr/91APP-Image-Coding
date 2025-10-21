custom_css = """
<style>
section.stMain {
    padding-left: 4%; 
    padding-right: 4%;
}
@media (min-width: 1900px) {
    section.stMain {
        padding-left: 11%;
        padding-right: 11%;
    }
}
@media (max-width: 1400px) {
    section.stMain {
        padding-left: 5%;
        padding-right: 5%;
    }
}
div.stTextInput > label {
    display: none;
}   
div.block-container {
    padding-top: 1.8rem;
}
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    padding: 5px 30px;
    background: #5A5B5E !important;
    color: #f5f5f5 !important;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    margin: 5px 0;
}
.stDownloadButton button {
    background: #5A5B5E !important;
    color: #f5f5f5 !important;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
[data-testid='stFileUploader'] section button {
    color: #46474A !important;
    border-radius: 5px;
    border: none;
    padding: 5px 40px;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #8A8B8D !important;
}
.stDownloadButton button:hover {
    background: #8A8B8D !important;
}
button:hover {
    background: #D3D3D3 !important;
}
div[data-testid=stToast] {
    background-color: #fff8b3;
}
header[data-testid="stHeader"] {
    height: 30px; 
    padding: 5px; 
}
[data-testid="stPopover"] {
    display: flex;
    justify-content: flex-end; 
}
[data-testid="stPopoverBody"] {
   width: 1000px !important;
}
[data-testid="stFileUploader"] [data-testid='stBaseButton-secondary'] {
  text-indent: -9999px;
  line-height: 0;
  margin-top: -10px !important;
  margin-bottom: -10px !important;
}
[data-testid="stFileUploader"] [data-testid='stBaseButton-secondary']::after {
    line-height: initial;
    content: "瀏覽檔案";
    text-indent: 0;
    font-size: 15px;
}
/* Drag and Drop File Here*/
[data-testid='stFileUploaderDropzoneInstructions'] > div > span {
  display: none;
}
/* file uploader 外框*/
[data-testid='stFileUploader'] section > input + div {
   height: 37px;
 }
[data-testid='stFileUploaderDropzoneInstructions'] > div::before {
  content: '上傳 ZIP 檔';
  font-size: 15px;
  margin-left: 8px !important;  
}
/* 上傳後的檔案圖示 hover*/
.eyeqlp53.st-emotion-cache-1b2ybts.ex0cdmw0 {
    width: 18px !important; 
    height: 18px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* 上傳後的檔案叉叉*/
.st-emotion-cache-clky9d {
    width: 24px !important;
    height: 24px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* 上傳後的檔案名稱文字 */
.stFileUploaderFileName {
    font-size: 15px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    width: 64%;
}
/* 淺色檔案大小文字 */
.st-emotion-cache-1aehpvj {
    font-size: 13!important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    margin-left: 8px !important;
}
/* 上傳後的檔案叉叉 hover*/
[data-testid="stFileUploaderDeleteBtn"] button {
    padding: 0px 10px !important; 
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* 上傳後的檔案名稱顯示*/
div[data-testid="stFileUploaderFile"] {
    margin-top: -66.5px;
    margin-left: -3px;
    width: 62%;
    background: #F0F2F6;
    border-radius: 5px;
    position: absolute; 
    z-index: 10;
    position: absolute;
    align-items: center;
}
/* 調整 st.divider 的上下空間 */
[data-testid="stMarkdownContainer"] hr {
    margin-top: 6px !important;
    margin-bottom: 14px !important;
    border-width: 1px !important;
}
[data-testid="stTextArea"] {
    margin-bottom: 15px; 
}
div[data-baseweb="select"] svg {
    display: none !important;
}
/* 調整選項字體大小 */
div[data-baseweb="select"] * {
    font-size: 15px !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stTooltipHoverTarget"] {
    font-size: 14px !important;
}
@media (min-width: 1900px) {
    div[data-testid="stTooltipHoverTarget"] {
        font-size: 15px !important;
    }
}
</style>
"""