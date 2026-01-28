import joblib
from sklearn.metrics import accuracy_score, classification_report
from utils.logger_config import logger

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.success(f"{model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc

def save_model(model, model_name):
    model_path = f"artifacts/{model_name}_model.joblib"
    joblib.dump(model, model_path)
    logger.success(f"{model_name} saved to {model_path}")