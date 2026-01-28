from data_processing.load_data import load_data
from data_processing.process_data import select_important_features, encode_data, process_and_split, apply_smote, save_data
from train.logistic_regression import train_logistic_regression
from train.random_forest import train_random_forest
from train.evaluate_model import evaluate_model, save_model

if __name__ == "__main__":
    
    # Load raw data
    df = load_data()

    # Select important features 
    df = select_important_features(df)

    # Encode data
    df_encoded = encode_data(df)

    # Split and scale data
    X_train, X_test, y_train, y_test = process_and_split(df_encoded)

    # Apply SMOTE to balance the training data
    X_train, y_train = apply_smote(X_train, y_train)

    # Save processed data
    save_data(X_train, X_test, y_train, y_test)

    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Save models
    save_model(lr_model, "logistic_regression")
    save_model(rf_model, "random_forest")