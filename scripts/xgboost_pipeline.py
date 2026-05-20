import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_pipeline.joblib")


def main():
    print("Hello from 188 Flight Delay project")

    # Load the dataset
    config.assert_data_exists()
    df = pd.read_csv(config.DATA_RAW / "Airlines.csv")

    # Clean column names (the CSV has leading/trailing spaces in headers)
    df.columns = df.columns.str.strip()

    # These columns have no effect on delay possibility
    columns_to_drop = [col for col in ["id", "Flight"] if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    # Target column
    target_column = "Delay"

    # Split into features and target
    X = df.drop(columns=target_column)
    y = df[target_column]

    categorical_cols = ["Airline", "AirportFrom", "AirportTo"]
    numeric_cols = ["DayOfWeek", "Time", "Length"]

    # Keep only columns that actually exist
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numeric_cols = [col for col in numeric_cols if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols),
        ]
    )

    # Feature selection: score all features with chi2, keep top k
    #chi2 function determines whether feature & target are independent of each other (flight feature vs delay)
    #k-factor is currently hard coded to 50, will be automated in GridSearchCV
    selector = SelectKBest(score_func=chi2, k=50) 

    # Full pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # GridSearchCV hyperparameter tuning
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "selector__k": [30, 50],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best ROC-AUC (CV): {grid_search.best_score_:.4f}")

    # Predict with best model
    y_pred = grid_search.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the best model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
    
