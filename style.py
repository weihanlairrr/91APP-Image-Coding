custom_css = """
<style>
section.stMain {
    padding-left: 12%; 
    padding-right: 12%;
}
@media (min-width: 1900px) {
    section.stMain {
        padding-left: 19%;
        padding-right: 19%;
    }
}
@media (max-width: 1400px) {
    section.stMain {
        padding-left: 8%;
        padding-right: 8%;
    }
}
div.stTextInput > label {
    display: none;
}   
div.block-container {
    padding-top: 2rem;
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
[data-testid="stFileUploader"] [data-testid='stBaseButton-secondary'] {
  text-indent: -9999px;
  line-height: 0;
}

[data-testid="stFileUploader"] [data-testid='stBaseButton-secondary']::after {
  line-height: initial;
  content: "瀏覽檔案";
  text-indent: 0;
  font-size: 15px;
}

[data-testid='stFileUploaderDropzoneInstructions'] > div > span {
  display: none;
}
[data-testid='stFileUploaderDropzoneInstructions'] > div::before {
  content: '將文件拖放到此處';
  font-size: 14px;
  margin-bottom: 3px; 
}
.eyeqlp53.st-emotion-cache-1b2ybts.ex0cdmw0 {
    width: 15px !important; 
    height: 15px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
.st-emotion-cache-clky9d {
    width: 22px !important;
    height: 22px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
.stFileUploaderFileName {
    font-size: 15px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
.st-emotion-cache-1aehpvj {
    font-size: 13px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
[data-testid="stFileUploaderDeleteBtn"] button {
    padding: 0px 10px !important; 
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* 調整 st.divider 的上下空間 */
[data-testid="stMarkdownContainer"] hr {
    margin-top: 6px !important;
    margin-bottom: 14px !important;
    border-width: 1px !important;
}

</style>
"""
