import streamlit as st
import pandas as pd
from Remarks_Predictorr import RemarksPredictor
from Remarks_Predictorr import TextCleaning
from Remarks_Predictorr import DataFramePredictor

def load_model(model_version):
    return RemarksPredictor(model_option=model_version)

def predict_with_model(model_version, input_text):
    model = load_model(model_version)
    return model.predict([input_text])

def render():
    st.title("BFI Collector Remarks Text Classification")

    # Inisialisasi session state untuk input teks dan model
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Naive Bayes"
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    # Input teks dari pengguna
    st.subheader("Input Remarks")
    st.session_state.input_text = st.text_area(
        "Enter remarks:", value=st.session_state.input_text, height=150
    )

    # Opsi memilih model NLP
    st.session_state.selected_model = st.selectbox(
        "Choose Model", ["Naive Bayes", "Random Forest", "SVM"],
        index=["Naive Bayes", "Random Forest", "SVM"].index(st.session_state.selected_model)
    )

    # Mapping pilihan model
    model_map = {
        "Naive Bayes": "MultinomialNB",
        "Random Forest": "RF",
        "SVM": "SVM"
    }

    # Tombol Submit
    if st.button("Submit"):
        if st.session_state.input_text.strip():
            try:
                prediction = predict_with_model(model_map[st.session_state.selected_model], st.session_state.input_text)
                st.session_state.prediction = prediction[0]
                st.success(f"Prediction ({st.session_state.selected_model}): {st.session_state.prediction}")
            except Exception as e:
                st.error(f"Error while predicting remarks: {str(e)}")
        else:
            st.warning("Enter a remarks first.")

    # Tombol Clear Input
    if st.button("Clear Input"):
        st.session_state.input_text = ""
        st.session_state.prediction = None

    # Data referensi untuk tabel
    data = {
        "Remarks": [
            "byr via PP.. Fu by wa",
            "kons tdk di rumah via tlfn masuk jb tgl 25",
            "visit ke tempat jualan PK di pasar tunggangri Kalidawir toko tertutup menghubungi nomor PK belum ada respon akan follow up ulang ",
            "visit ke tempat kerja kpns tdk ketemu info dari rekan kons ddg ikut kegiatan di kota",
            "visit ke almt saudara gali info tentang kepemilikan unit info dr saudara nya itu bukan punyak nya dan tdk tau karena unit drmh hanya ada beat dan vega.",
            "visit ke almt lain bertemu.jb max besok.fu ulang.kendala smc waktu ktmu deb tidak muncul customer list.",
            "visit ke kos2an tdk ketemu kons"
        ]
    }
    df = pd.DataFrame(data)

    # Header untuk Referensi Data
    st.subheader("Examples")
    st.markdown("#### Remarks Examples")

    # Fungsi untuk memperbarui teks di text area
    def update_text_area(text):
        st.session_state.input_text = text

    # Tampilkan data referensi
    for i, row in df.iterrows():
        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"<p style='margin-bottom: 0;'>{row['Remarks']}</p>", unsafe_allow_html=True)
        with col2:
            st.button("Select", key=f"button_{i}", on_click=update_text_area, args=(row["Remarks"],))
        st.markdown("<hr style='border: 0; height: 1px; background-color: #ddd; margin-top: 0; margin-bottom: 2px;' />", unsafe_allow_html=True)