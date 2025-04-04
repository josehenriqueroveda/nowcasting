import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)


def build_features():
    config = load_config()
    df = pd.read_feather(config["data"]["processed_path"])

    X_res, y_res = df[config["model"]["features"]], df["target_chuva"]

    scaler = RobustScaler()
    cols_to_scale = [col for col in X_res.columns if col != "hora"]
    X_res[cols_to_scale] = scaler.fit_transform(X_res[cols_to_scale])

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=config["data"]["test_size"], shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler
