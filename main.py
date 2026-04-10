import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def main():
    print("Hello from 188 Flight Delay project")

    # Load the dataset
    df = pd.read_csv("Airlines.csv")

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

    # missing feature selection implementation (left for other teammates)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols),
        ]
    )

    # Full pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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

    # Missing GridSearch hyperparameter tuning (left for other teammembers) implementation
    
    
    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
    
