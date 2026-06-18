import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
def load_data():
    """Load dataset"""
    df = pd.read_csv("Iris.csv")
    return df
def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n===== DATASET OVERVIEW =====")
    print(df.head())
    print("\nDataset Shape:")
    print(df.shape)
    print("\nColumn Names:")
    print(df.columns)
    print("\nDataset Information:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
def visualize_data(df):
    """Visualize dataset"""
    sns.pairplot(df, hue="Species")
    plt.show()
def train_model(df):
    """Train KNN model"""

    X = df[['SepalLengthCm',
            'SepalWidthCm',
            'PetalLengthCm',
            'PetalWidthCm']]
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n===== MODEL EVALUATION =====")
    print("Accuracy:",
          round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return model, scaler
def predict_species(model, scaler):
    """Predict new flower species"""
    print("\n===== PREDICT NEW FLOWER =====")
    sepal_length = float(input("Enter Sepal Length: "))
    sepal_width = float(input("Enter Sepal Width: "))
    petal_length = float(input("Enter Petal Length: "))
    petal_width = float(input("Enter Petal Width: "))
    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]
    sample = scaler.transform(sample)
    prediction = model.predict(sample)
    print("\nPredicted Species:", prediction[0])
def main():
    df = load_data()
    explore_data(df)
    visualize_data(df)
    model, scaler = train_model(df)
    predict_species(model, scaler)
if __name__ == "__main__":
    main()