from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import shap

def avaliar_modelo(y_test, y_pred):
    print(classification_report(y_test, y_pred))

def plot_roc_curve(y_test, y_probs):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.show()
    print("AUC:", roc_auc_score(y_test, y_probs))

def explicar_shap(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])
    shap.plots.bar(shap_values)