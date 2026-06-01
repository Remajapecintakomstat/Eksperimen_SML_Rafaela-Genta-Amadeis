
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    return df


def clean_data(df):
    """
    Data cleaning
    """
    # Drop customerID
    df.drop(columns=["customerID"], inplace=True)

    # Handle blank values
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)

    # Convert datatype
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df.dropna(inplace=True)

    print(f"After cleaning: {df.shape}")

    return df


def encode_data(df):
    """
    Encoding target and categorical features
    """

    # Encode target
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # One-hot encoding
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    print(f"After encoding: {df.shape}")

    return df


def split_data(df):
    """
    Split features and target
    """

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
    Standardization
    """

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(
        X_train[numeric_cols]
    )

    X_test[numeric_cols] = scaler.transform(
        X_test[numeric_cols]
    )

    return X_train, X_test


def save_data(X_train, X_test, y_train, y_test):
    """
    Save processed dataset
    """

    os.makedirs(
        "dataset_preprocessing",
        exist_ok=True
    )

    train_df = pd.concat(
        [X_train, y_train],
        axis=1
    )

    test_df = pd.concat(
        [X_test, y_test],
        axis=1
    )

    train_df.to_csv(
        "dataset_preprocessing/train_processed.csv",
        index=False
    )

    test_df.to_csv(
        "dataset_preprocessing/test_processed.csv",
        index=False
    )

    print("Processed dataset saved.")


def main():

    filepath = "Telco Customer Churn_raw.csv"

    df = load_data(filepath)

    df = clean_data(df)

    df = encode_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test = scale_data(
        X_train,
        X_test
    )

    save_data(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()
