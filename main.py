from src.data_preparacao import carregar_e_preparar_dados
from src.modelos import treinar_xgboost
from src.avaliacao import avaliar_modelo

# 1. Preparar dados
X_train, X_test, y_train, y_test = carregar_e_preparar_dados()

# 2. Treinar
modelo_xgb = treinar_xgboost(X_train, y_train)

# 3. Prever e Avaliar
previsoes = modelo_xgb.predict(X_test)
avaliar_modelo(y_test, previsoes)