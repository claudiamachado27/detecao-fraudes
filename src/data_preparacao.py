import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def carregar_e_preparar_dados():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    
    # Engenharia de variáveis
    df["Amount_log"] = np.log1p(df["Amount"])
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    
    # Split
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)