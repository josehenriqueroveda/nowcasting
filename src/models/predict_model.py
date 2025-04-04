import joblib
import pandas as pd
from pathlib import Path

# Configuração das colunas que devem ser escaladas (deve corresponder ao treinamento)
COLS_TO_SCALE = [
    "Temp. Ins. (C)",
    "Umi. Ins. (%)",
    "Pto Orvalho Ins. (C)",
    "Pressao Ins. (hPa)",
    "Vel. Vento (m/s)",
    "Raj. Vento (m/s)",
    "pressao_media_3h",
    "variacao_pressao_1h",
    "delta_temp_3h",
    "vento_x",
    "vento_y",
]

# Exemplo de dados da última hora observada
LAST_HOUR_DATA = pd.DataFrame(
    [
        {
            "Temp. Ins. (C)": 18.5,
            "Umi. Ins. (%)": 84.0,
            "Pto Orvalho Ins. (C)": 15.6,
            "Pressao Ins. (hPa)": 886.2,
            "Vel. Vento (m/s)": 2.2,
            "Raj. Vento (m/s)": 4.2,
            "hora": 3,
            "pressao_media_3h": 886.3,
            "variacao_pressao_1h": -0.4,
            "delta_temp_3h": 0.7,
            "vento_x": 0.5,
            "vento_y": 2.1,
        }
    ]
)


def predict(last_data: pd.DataFrame = LAST_HOUR_DATA):
    """Faz a predição de chuva para a próxima hora.

    Args:
        last_data (pd.DataFrame): DataFrame com os dados da última hora observada.
                                 Deve conter as mesmas colunas usadas no treinamento.
    """
    try:
        # Carrega modelo e scaler
        models_dir = Path("models/trained/")
        model = joblib.load(models_dir / "lgbm_model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")

        # Garante a ordem correta das colunas
        last_data = last_data[model.feature_name_]

        # Aplica a mesma escala usada no treinamento
        last_data_scaled = last_data.copy()
        last_data_scaled[COLS_TO_SCALE] = scaler.transform(last_data[COLS_TO_SCALE])

        # Faz a predição
        proba = model.predict_proba(last_data_scaled)[0][1]

        # Exibe o resultado
        print(f"\nProbabilidade de chuva na próxima hora: {proba:.1%}")
        print("Status:", "Chuva prevista" if proba > 0.5 else "Sem previsão de chuva")

        return proba

    except Exception as e:
        print(f"Erro na predição: {str(e)}")
        raise
