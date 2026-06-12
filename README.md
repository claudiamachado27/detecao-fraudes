# 🔍 Detecção de Fraude em Cartão de Crédito

Projeto de Machine Learning para detecção de transações fraudulentas em cartões de crédito, utilizando técnicas de classificação supervisionada com dados altamente desbalanceados.

## 📋 Sobre o Projeto

Este projeto treina e avalia modelos de ML para identificar fraudes em transações financeiras com base no [dataset público do Kaggle / TensorFlow](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv). O dataset contém transações reais anonimizadas de cartões de crédito europeus, com apenas ~0,17% de transações fraudulentas.

### Modelos Implementados

| Modelo | Descrição |
|---|---|
| **Logistic Regression** | Baseline linear |
| **Random Forest** | Ensemble com balanceamento de classes |
| **XGBoost** | Gradient boosting com `scale_pos_weight` para lidar com desbalanceamento |

## 🗂️ Estrutura do Projeto

```
projeto-python/
├── main.py                  # Ponto de entrada principal
├── requirements.txt         # Dependências do projeto
├── data/                    # Pasta para dados locais (vazia — dados carregados via URL)
├── notebooks/               # Notebooks Jupyter (exploração e experimentos)
└── src/
    ├── data_preparacao.py   # Carregamento, feature engineering e split
    ├── modelos.py           # Treino dos modelos (LR, RF, XGBoost)
    └── avaliacao.py         # Avaliação, curva ROC e explicabilidade SHAP
```

## ⚙️ Requisitos

- Python 3.13+
- macOS: `libomp` instalado via Homebrew (necessário para XGBoost)

```bash
brew install libomp
```

## 🚀 Como Executar

### 1. Clonar e entrar na pasta

```bash
git clone <url-do-repositório>
cd projeto-python
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar

```bash
python main.py
```

> **Nota:** Na primeira execução o script faz o download automático do dataset (~143 MB). Aguarde alguns segundos.

## 📦 Dependências

```
pandas
numpy
scikit-learn
xgboost
matplotlib
shap
imblearn
```

## 📊 Pipeline de Execução

```
1. Carregar dados         → data_preparacao.py
      ↓
2. Feature Engineering    → log(Amount), StandardScaler
      ↓
3. Train/Test Split       → 70% treino / 30% teste, stratificado
      ↓
4. Treinar XGBoost        → modelos.py
      ↓
5. Avaliar modelo         → avaliacao.py
      ├── Classification Report (precision, recall, F1)
      ├── Curva ROC + AUC
      └── Explicabilidade SHAP (top 100 amostras)
```

## 📈 Funcionalidades de Avaliação

- **`avaliar_modelo(y_test, y_pred)`** — imprime o classification report completo
- **`plot_roc_curve(y_test, y_probs)`** — plota a curva ROC e calcula AUC
- **`explicar_shap(model, X_test)`** — gera gráfico de importância de features via SHAP

## 🔧 Resolução de Problemas

### `xgboost.core.XGBoostError: libxgboost.dylib could not be loaded`

```bash
brew install libomp
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /opt/homebrew/lib/libomp.dylib
```

### `command not found: python`

No macOS use `python3` ou ative o ambiente virtual (`source venv/bin/activate`).

### Erros de módulo no IDE (pylance/pyright)

Selecione o interpretador Python do `venv` no IDE:
```
/caminho/do/projeto/venv/bin/python
```
No VS Code / Cursor: `Cmd+Shift+P` → **Python: Select Interpreter**
