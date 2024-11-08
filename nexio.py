import sys
import requests
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print(f"Python {sys.version}")

import subprocess
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

customers_data = pd.DataFrame({
    'Client_ID': [101, 102, 103],
    'Name': ["Alice", "Bob", "Charlie"],
    'Stage': ["Lead", "Client", "Prospect"]
})

def update_customer_stage(client_id, new_stage):
    customers_data.loc[customers_data['Client_ID'] == client_id, 'Stage'] = new_stage
    print(f"Cliente {client_id} actualizado a la etapa: {new_stage}")

def follow_up(client_id):
    client_info = customers_data[customers_data['Client_ID'] == client_id]
    if not client_info.empty:
        print(f"Enviando seguimiento a {client_info['Name'].values[0]}.")
    else:
        print("Cliente no encontrado.")

def handle_query(query):
    query_vector = nlp(query).vector
    responses = ["Para soporte técnico, visita nuestra sección de ayuda.",
                 "Para consultas comerciales, contacta a nuestro equipo de ventas."]
    similarities = [cosine_similarity([query_vector], [nlp(response).vector])[0][0] for response in responses]
    best_response = responses[similarities.index(max(similarities))]
    return best_response

if __name__ == "__main__":
    update_customer_stage(101, "Client")
    follow_up(101)
    query = "¿Cómo puedo contactar al equipo de ventas?"
    print("Respuesta del bot:", handle_query(query))
