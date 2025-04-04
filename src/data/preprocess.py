import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)


def preprocess_data():
    config = load_config()
    df = pd.read_csv(
        config["data"]["raw_path"],
        sep=";",
        decimal=",",
        parse_dates={"timestamp": ["Data", "Hora (UTC)"]},
        dayfirst=True,
    )
    df.set_index("timestamp", inplace=True)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    df["target_chuva"] = (df["Chuva (mm)"].shift(-1) > 0).astype(int)
    df.dropna(subset=["target_chuva"], inplace=True)

    df["hora"] = df.index.hour
    df["pressao_media_3h"] = df["Pressao Ins. (hPa)"].rolling(3).mean()
    df["variacao_pressao_1h"] = df["Pressao Ins. (hPa)"].diff(1)
    df["delta_temp_3h"] = df["Temp. Ins. (C)"].diff(3)

    df["vento_x"] = df["Vel. Vento (m/s)"] * np.sin(np.radians(df["Dir. Vento (m/s)"]))
    df["vento_y"] = df["Vel. Vento (m/s)"] * np.cos(np.radians(df["Dir. Vento (m/s)"]))

    df["Radiacao (KJ/m²)"] = df["Radiacao (KJ/m²)"].fillna(
        df.groupby(df.index.hour)["Radiacao (KJ/m²)"].transform("mean")
    )

    df = df[config["model"]["features"] + ["target_chuva"]]
    df.fillna(method="ffill", inplace=True)

    # Salvar
    Path(config["data"]["processed_path"]).parent.mkdir(exist_ok=True)
    df.to_feather(config["data"]["processed_path"])


if __name__ == "__main__":
    preprocess_data()
