import joblib
import pandas as pd

# Load the trained model
model = joblib.load('loan_approval_model.pkl')

# Variables need to predict
features = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}
}


# Function to take customers data and predict if loan is aproved or not
def predict_loan_status(data_dic):
    try:
        # Map categorical variables
        for key in features:
            if key in data_dic:
                data_dic[key] = features[key][data_dic[key]]

        # Convert to DataFrame
        df = pd.DataFrame([data_dic])

        # Make prediction
        prediction = model.predict(df)

        return "Aprobado" if prediction[0] == 1 else "Rechazado"

    except Exception as e:
        return f"Error en la predicción: {e}"
