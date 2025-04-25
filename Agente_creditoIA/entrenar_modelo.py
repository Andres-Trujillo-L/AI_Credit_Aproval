import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cargar dataset
train_df = pd.read_csv('train2.csv')

# Imputar valores
cat_cols = ['Gender', 'Married', 'Self_Employed', 'Dependents', 'Loan_Amount_Term', 'Credit_History']
num_cols = ['LoanAmount']

cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')

train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])

# Codificar variables categóricas
label_encoders = {}
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# Etiqueta
train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})
train_df = train_df.dropna(subset=['Loan_Status'])

# Variables de entrada
features = ['Gender', 'Married', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Dependents']
X = train_df[features]
y = train_df['Loan_Status']

# Entrenar modelo
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model, 'loan_approval_model.pkl')