import streamlit as st
import homepage_final
import dfpage_final
from Remarks_Predictorr import RemarksPredictor
from Remarks_Predictorr import TextCleaning
from Remarks_Predictorr import DataFramePredictor

# Sidebar untuk opsi halaman
st.sidebar.markdown("### Navigation")
page = st.sidebar.selectbox("Go to:", ["Homepage", "DataFrame Classification"])

# Render halaman sesuai pilihan
if page == "Homepage":
    homepage_final.render()
elif page == "DataFrame Classification":
    dfpage_final.render()

