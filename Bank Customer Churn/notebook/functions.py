import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def plot_boxplots(data):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(y=data['credit_score'], ax=axs[0], color='skyblue')
    axs[0].set_title('Boxplot of Credit Score')
    axs[0].set_yticks(axs[0].get_yticks())
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=90)

    sns.boxplot(y=data['balance'], ax=axs[1], color='salmon')
    axs[1].set_title('Boxplot of Balance')
    axs[1].set_yticks(axs[1].get_yticks())
    axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=90)

    sns.boxplot(y=data['estimated_salary'], ax=axs[2], color='green')
    axs[2].set_title('Boxplot of Estimated Salary')
    axs[2].set_yticks(axs[2].get_yticks())
    axs[2].set_yticklabels(axs[2].get_yticklabels(), rotation=90)

    plt.tight_layout()
    plt.show()

def plot_active_member_distribution(data):
    active_counts = data['active_member'].value_counts()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    axs[0].pie(active_counts, labels=['Inactive', 'Active'], autopct='%1.1f%%', colors=['lightcoral', 'skyblue'], startangle=140)
    axs[0].set_title('Distribution of Active Members')
    axs[0].axis('equal')

    # Bar chart
    sns.countplot(x='active_member', data=data, palette=['lightcoral', 'skyblue'], ax=axs[1])
    for p in axs[1].patches:
        height = p.get_height()
        axs[1].annotate(f'{height}', xy=(p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axs[1].set_title('Count of Active vs Inactive Members')
    axs[1].set_xlabel('Active Member')
    axs[1].set_ylabel('Count')
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(['Inactive', 'Active'])

    plt.tight_layout()
    plt.show()

def plot_country_distribution(data):
    country_counts = data['country'].value_counts()
    unique_countries = country_counts.index

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    axs[0].pie(country_counts, labels=unique_countries, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel')[:len(unique_countries)])
    axs[0].set_title('Distribution of Countries')
    axs[0].axis('equal')

    # Bar chart
    sns.countplot(x='country', data=data, palette=sns.color_palette('pastel')[:len(unique_countries)], ax=axs[1])
    for p in axs[1].patches:
        height = p.get_height()
        axs[1].annotate(f'{height}', xy=(p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axs[1].set_title('Count of Countries')
    axs[1].set_xlabel('Country')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()
    

def balance_age(data):
    plt.figure(figsize=(10, 6))
    # scatter plot
    sns.scatterplot(x='age', y='balance', data=data, hue='gender', palette='coolwarm')
    plt.title('Balance by Age')
    plt.xlabel('Age')
    plt.ylabel('Balance')
    plt.show()

def select_features_and_target(data, target_column):

    x = data.drop(columns=["customer_id", target_column])
    y = data[target_column]
    return x,y

def get_numerics_col(data):
    numerics = ["int32", "int64", "float32", "float64"]
    numerics_x = data.select_dtypes(include=numerics)
    numerics_col = list(numerics_x.columns)
    return numerics_col

def get_categorical_col(data, numeric_cols):
    features = data.columns
    categorical_cols = list(set(features) - set(numeric_cols))
    return categorical_cols

def train_val_test_split(X, Y, test_size=0.1, val_size=0.1, random_state=42): 
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=test_size + val_size, random_state=random_state, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=Y_temp)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def scale_numeric_data(X_train, X_val, X_test, numeric_columns):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_val_scaled[numeric_columns] = scaler.transform(X_val[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    return X_train_scaled, X_val_scaled, X_test_scaled


def encode_categorical_data(X_train, X_val, X_test, categorical_columns):
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Fit the encoder on the training data
    encoder.fit(X_train[categorical_columns])
    
    # Transform the data
    X_train_encoded = encoder.transform(X_train[categorical_columns])
    X_val_encoded = encoder.transform(X_val[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])
    
    return X_train_encoded, X_val_encoded, X_test_encoded
