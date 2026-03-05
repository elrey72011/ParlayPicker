import pandas as pd


def load_theover_csv(file):
    if file is None:
        return None

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    return df
