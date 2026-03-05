import pandas as pd
import streamlit as st


def load_theover_csv(uploaded_file):

    if uploaded_file is None:
        return pd.DataFrame()

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("Uploaded TheOver CSV is empty.")
            return pd.DataFrame()

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        return df

    except pd.errors.EmptyDataError:

        st.warning("Uploaded CSV contains no readable data.")
        return pd.DataFrame()

    except Exception as e:

        st.error(f"CSV loading error: {e}")
        return pd.DataFrame()
