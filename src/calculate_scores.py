import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Función para calcular el score de Importancia
def calculate_importancy(description):
    """
    Calcula el score de importancia de un evento basado en su descripción textual.
    Esta función debe integrarse con un modelo de lenguaje para evaluar la importancia.
    
    Args:
    description (str): Descripción del evento.
    
    Returns:
    int: Puntaje de importancia del evento.
    """
    # Llamada simulada a un modelo de lenguaje, reemplazar con una llamada real a la API de GPT
    importance_score = simulate_language_model_prediction(description)
    return importance_score

def simulate_language_model_prediction(description):
    """
    Simula la predicción de un modelo de lenguaje para la importancia de un evento.
    En un caso de uso real, esta función debería conectarse a un API de GPT que pueda evaluar descripciones de eventos.
    
    Args:
    description (str): Descripción del evento que se evaluará.
    
    Returns:
    int: Puntaje simulado de importancia.
    """
    # Aquí se debería implementar una llamada a la API de GPT, con un prompt basado en la descripción.
    # Por ejemplo: "On a scale from 1 to 10, how important is this event: {description}?"
    return 8  # Valor simulado, debería ser reemplazado por la respuesta de la API de GPT

# Función para calcular el score de Recency
def calculate_recency(last_access_date):
    """
    Calcula el score de recency, indicando cuán reciente fue el acceso a un evento.
    
    Args:
    last_access_date (datetime): La fecha del último acceso al evento.
    
    Returns:
    float: Score de recency, normalizado entre 0 y 1.
    """
    today = datetime.datetime.now()
    recency_days = (today - last_access_date).days
    # Normalización para mantener el score entre 0 y 1
    return max(0, 1 - (recency_days / 365))  # Supone que después de un año, el evento no tiene relevancia

# Función para calcular el score de Relevance
def calculate_relevance(documents, query):
    """
    Calcula la relevancia de un conjunto de documentos con respecto a una consulta usando TF-IDF y similitud del coseno.
    
    Args:
    documents (list of str): Lista de documentos o textos de referencia.
    query (str): Texto de la consulta para comparar con los documentos.
    
    Returns:
    numpy.ndarray: Array de scores de relevancia para cada documento en relación con la consulta.
    """
    vectorizer = TfidfVectorizer()
    # Transforma los documentos y la consulta en vectores TF-IDF
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    # Calcula la similitud del coseno entre la consulta y cada documento
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return cosine_similarities.flatten()

# Ejemplo de uso de las funciones
description_event = "Breakup with a significant other"
importancy_score = calculate_importancy(description_event)

last_access_date = datetime.datetime.strptime('2022-12-01', '%Y-%m-%d')
recency_score = calculate_recency(last_access_date)

documents = ["the cat in the hat", "a quick brown fox jumps over the lazy dog", "machine learning models"]
query = "text about a cat and a dog"
relevance_scores = calculate_relevance(documents, query)

print(f"Importancy Score: {importancy_score}")
print(f"Recency Score: {recency_score}")
print(f"Relevance Scores: {relevance_scores}")

