import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def select_features_and_target(data, target_column):
    x = data.drop(columns=["customer_id", target_column])
    y = data[target_column]
    return x, y

def get_numerics_col(data):
    numerics = ["int32", "int64", "float32", "float64"]
    numerics_x = data.select_dtypes(include=numerics)
    numerics_col = list(numerics_x.columns)
    return numerics_col

def get_categorical_col(data, numeric_cols):
    features = data.columns
    categorical_cols = list(set(features) - set(numeric_cols))
    return categorical_cols

def scale_numeric_data(X_train, X_test, numeric_columns):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    return X_train_scaled, X_test_scaled

def encode_categorical_data(X_train, X_test, categorical_columns):
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Fit the encoder on the training data
    encoder.fit(X_train[categorical_columns])
    
    # Transform the data
    X_train_encoded = encoder.transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])
    
    # Convert the encoded data back to DataFrame for easier concatenation
    X_train_encoded_df = pd.DataFrame(X_train_encoded, index=X_train.index, columns=encoder.get_feature_names_out(categorical_columns))
    X_test_encoded_df = pd.DataFrame(X_test_encoded, index=X_test.index, columns=encoder.get_feature_names_out(categorical_columns))
    
    return X_train_encoded_df, X_test_encoded_df

def preprocess_data(X_train, X_test, numeric_columns, categorical_columns):
    # Scale numeric data
    X_train_scaled, X_test_scaled = scale_numeric_data(X_train, X_test, numeric_columns)
    
    # Encode categorical data
    X_train_encoded, X_test_encoded = encode_categorical_data(X_train, X_test, categorical_columns)
    
    # Concatenate scaled numeric data and encoded categorical data
    X_train_preprocessed = pd.concat([X_train_scaled, X_train_encoded], axis=1)
    X_test_preprocessed = pd.concat([X_test_scaled, X_test_encoded], axis=1)
    
    return X_train_preprocessed, X_test_preprocessed
