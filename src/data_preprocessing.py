# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(
    file_path,
    test_size=0.2,
    random_state=42
):
    """
    Loads the credit card dataset, preprocesses it,
    and returns scaled train-test splits.

    Parameters:
    ----------
    file_path : str
        Path to the dataset file (.xls)
    test_size : float
        Proportion of test data
    random_state : int
        Seed for reproducibility

    Returns:
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """

    # Load dataset
    df = pd.read_excel(file_path, header=1)

    # Drop non-informative column
    df = df.drop(columns=["ID"])

    # Separate features and target
    X = df.drop("default payment next month", axis=1)
    y = df["default payment next month"]

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
