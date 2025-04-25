# AI_Credit_Aproval

Este proyecto implementa un agente inteligente basado en LangChain que decide si un crédito debe ser aprobado o no, utilizando un modelo de machine learning previamente entrenado.

## ¿Qué hace este agente?
Recibe datos de un solicitante de crédito en formato JSON.

Interpreta la información (usando OpenAI o Hugging Face).

Llama a una función predict_loan_status() que hace la predicción.

### Devuelve una respuesta del tipo:

  ***-Crédito aprobado***
  ***-Crédito rechazado***

## Cómo usar

### 1. Clona este repositorio

git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio

### 2. Instala dependencias

pip install -r requirements.txt

O manualmente:

pip install langchain openai joblib pandas python-dotenv

### 3. Configura tu API Key de OpenAI

Crea un archivo .env en la raíz del proyecto y añade:

OPENAI_API_KEY=sk-tu-clave-aqui
Sin comillas

### 4. Ejecuta el agente

python agente.py

Ingresa los datos del crédito en formato JSON:

{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360.0,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
Respuesta esperada:

### Resultado del crédito: Crédito aprobado

Estructura del proyecto

Agente_creditoIA/
│
├── entrenar_modelo.py         # entrenas y guardas el modelo
├── predictor.py               # función para predicciones individuales
├── evaluacion.py              # evaluación objetiva con test.csv
├── modelo/
│   ├── loan_approval_model.pkl
│   ├── agent.py
├── data/
│   ├── train.csv
│   ├── test.csv
└── requirements.txt           # librerías necesarias

Requisitos del sistema
Python 3.8+

### En desarrollo futuro
  -Interfaz web con Streamlit
  -Integración con base de datos
  -API REST con FastAPI

#  Autor
Desarrollado por Andrés Trujillo como parte de su portafolio de Data Science y Python Development.

📫 [LinkedIn](https://www.linkedin.com/in/andres-trujillo-luzuriaga) | 🌐 [GitHub](https://github.com/Andres-Trujillo-L)
