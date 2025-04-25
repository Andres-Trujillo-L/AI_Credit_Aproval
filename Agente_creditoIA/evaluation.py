import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar modelo y datos
model = joblib.load('modelo/loan_approval_model.pkl')
test_df = pd.read_csv('data/test.csv')

# Imputación y transformación de test.csv igual que en entrenamiento
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Gender', 'Married', 'Self_Employed', 'Dependents', 'Loan_Amount_Term', 'Credit_History']
num_cols = ['LoanAmount']

cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')

test_df[cat_cols] = cat_imputer.fit_transform(test_df[cat_cols])
test_df[num_cols] = num_imputer.fit_transform(test_df[num_cols])

for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])

# Variables
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area']

# Predicción
predictions = model.predict(test_df[features])
test_df['Predicted'] = predictions
test_df['Predicted_Label'] = np.where(predictions == 1, 'Y', 'N')

# Resultados
print("📌 Precisión:", accuracy_score(test_df['Loan_Status'], predictions))
print("\n📌 Reporte de Clasificación:\n", classification_report(test_df['Loan_Status'], predictions))

# Matriz de confusión
cm = confusion_matrix(test_df['Loan_Status'], predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.tight_layout()
plt.show()
