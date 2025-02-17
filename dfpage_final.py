import streamlit as st
import pandas as pd
from Remarks_Predictorr import DataFramePredictor


def load_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        file_df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip', sep=';')
    else:
        file_df = pd.read_excel(uploaded_file)

    if 'remarks' not in file_df.columns:
        return None

    file_df['remarks'] = file_df['remarks'].fillna('')
    return file_df

def process_subset_data(subset_df):
    dfpredictor = DataFramePredictor()
    predicted_subset_df = dfpredictor.dfpredict(subset_df)  # Run model prediction
    return predicted_subset_df

def render():
    st.title("BFI Collector Remarks Text Classification")
    st.subheader("DataFrame Classification")

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    st.session_state['uploaded_file'] = uploaded_file
    if uploaded_file is not None:
        file_df = load_file(uploaded_file)  # Load data tanpa prediksi

        if file_df is None:
            st.error("Kolom 'remarks' tidak ditemukan dalam data.")
            return

        total_items = len(file_df)
        items_per_page = 200
        total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)

        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 1

        st.session_state['current_page'] = st.selectbox(
            "Select Page", list(range(1, total_pages + 1)),
            index=st.session_state['current_page'] - 1
        )

        start_idx = (st.session_state['current_page'] - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        subset_df = file_df.iloc[start_idx:end_idx].copy()  # Ambil subset data untuk halaman ini

        if 'predicted_subset_df' not in st.session_state or st.session_state['last_page'] != st.session_state['current_page']:
            st.session_state['predicted_subset_df'] = process_subset_data(subset_df)  # Prediksi hanya untuk halaman ini
            st.session_state['last_page'] = st.session_state['current_page']

        paginated_df = st.session_state['predicted_subset_df']
        paginated_df.index = range(start_idx + 1, start_idx + len(paginated_df) + 1)

        st.write(f"Showing {start_idx + 1}-{end_idx} of {total_items} records (Page {st.session_state['current_page']} of {total_pages})")
        st.dataframe(paginated_df)

        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 6, 1, 1, 1, 1])
        with col1:
            if st.button("⏮ First", key="first"):
                st.session_state['current_page'] = 1
                st.rerun()
        with col2:
            if st.button("◀ Prev", key="prev") and st.session_state['current_page'] > 1:
                st.session_state['current_page'] -= 1
                st.rerun()
        with col6:
            if st.button("Next ▶", key="next") and st.session_state['current_page'] < total_pages:
                st.session_state['current_page'] += 1
                st.rerun()
        with col7:
            if st.button("Last ⏭", key="last"):
                st.session_state['current_page'] = total_pages
                st.rerun()

        csv = paginated_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download data as CSV", data=csv, file_name="predicted_data.csv", mime="text/csv")

    else:
        st.write("No file uploaded yet.")

if __name__ == "__main__":
    render()