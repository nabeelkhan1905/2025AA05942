import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load data
df = pd.read_csv('data/bank-full.csv', sep=';')

# Preprocessing
data = df.copy()
data['y'] = data['y'].map({'yes': 1, 'no': 0})
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

X = data.drop('y', axis=1)
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save Scaler (Very important for app.py consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))

# Define and Train the Big 6
models = {
    "logistic_regression": LogisticRegression(),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier()
}

for name, model in models.items():
    if name in ["logistic_regression", "knn"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Save each model
    pickle.dump(model, open(f'model/{name}.pkl', 'wb'))
    print(f"Saved {name}.pkl")