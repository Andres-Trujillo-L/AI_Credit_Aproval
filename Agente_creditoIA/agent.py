import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from predictor import predict_loan_status

# Agent personalized tool
def analizar_credito(input_text: str) -> str:
    """
    Wait for a JSON string with the data.
    """
    import json
    try:
        data = json.loads(input_text)
        resultado = predict_loan_status(data)
        return f"Resultado del crédito: {resultado}"
    except Exception as e:
        return f"Error al analizar los datos del crédito: {str(e)}"

# Make LangChain tool
tools = [
    Tool(
        name="AnalizarCrédito",
        func=analizar_credito,
        description="Usa esta herramienta para decidir si un crédito debe ser aprobado. Recibe un JSON con datos como 'Gender', 'Married', 'ApplicantIncome', etc."
    )
]

# Init the LLM and the agent
llm = OpenAI(temperature=0)
agente = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ejemplo de uso
if __name__ == "__main__":
    print("🤖 Agente de análisis crediticio")
    print("Introduce los datos en formato JSON (o escribe 'salir')")

    while True:
        entrada = input("👉 ")
        if entrada.lower() == "salir":
            break
        respuesta = agente.run(f"Evalúa este crédito: {entrada}")
        print("🔎", respuesta)
