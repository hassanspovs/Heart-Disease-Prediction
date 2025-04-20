# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\hassa\OneDrive\Documents\Visual Studio 2022\ML PROJECTS\heart.csv")

print("Dataset Head:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Handle missing values (if any)
df = df.dropna()  # Dropping rows with missing values

X = df.drop('target', axis=1)  # Features (all columns except target)
y = df['target']  # Target column (label to predict)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'heart_disease_model.pkl')
print("\nModel saved as 'heart_disease_model.pkl'")

