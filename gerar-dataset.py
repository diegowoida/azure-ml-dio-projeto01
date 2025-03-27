import pandas as pd
import numpy as np

# Criar datas de 2024
dates = pd.date_range(start="2024-01-01", end="2024-12-31")

# Simular vendas e temperatura com sazonalidade
np.random.seed(42)
vendas = np.where(
    dates.month.isin([12, 1, 2]),  # Verão
    np.random.randint(100, 150, len(dates)),
    np.where(
        dates.month.isin([9, 10, 11]),  # Primavera
        np.random.randint(90, 130, len(dates)),
        np.where(
            dates.month.isin([6, 7, 8]),  # Inverno
            np.random.randint(40, 70, len(dates)),
            np.random.randint(60, 90, len(dates))  # Outono
        )
    )
)

# Simular temperatura (valores médios por estação)
temp = np.where(
    dates.month.isin([12, 1, 2]), 28 - (dates.day / 10),
    np.where(
        dates.month.isin([3, 4, 5]), 22 - (dates.month - 3),
        np.where(
            dates.month.isin([6, 7, 8]), 12 + (dates.month - 6),
            18 + (dates.month - 9)  # Primavera
        )
    )
)

# Criar DataFrame
df = pd.DataFrame({
    "Data": dates,
    "Vendas": vendas,
    "Temperatura": np.round(temp, 1),
    "Estação": np.select(
        [dates.month.isin([12, 1, 2]), dates.month.isin([3, 4, 5]), dates.month.isin([6, 7, 8])],
        ["Verão", "Outono", "Inverno"],
        default="Primavera"
    )
})

# Salvar em CSV
df.to_csv("vendas_temperatura_2024.csv", index=False)