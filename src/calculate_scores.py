import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Observation:
    def __init__(self, description, importancy=None, relevance=None, recency=None, timestamp=None, last_access_date=None):
        self._description = description
        self._importancy = importancy
        self._relevance = relevance
        self._recency = recency
        self._timestamp = timestamp
        self._last_access_date = last_access_date

    def __del__(self):
        pass

# Función para calcular el score de Importancia
@property
def importancy(self):
    return self._importancy

@importancy.setter
def importancy(self):
    """
    Calcula el score de importancia de un evento basado en su descripción textual usando una API de GPT.
    
    Args:
    description (str): Descripción del evento.
    
    Returns:
    int: Puntaje de importancia del evento.
    """
    url = 'https://api.openai.com/v1/engines/davinci/completions'  # URL de la API de OpenAI GPT
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',  # Sustituye 'YOUR_API_KEY' con tu clave API real
        'Content-Type': 'application/json'
    }
    data = {
        'prompt': f'On a scale from 1 to 10, how important is this event: {description}?',
        'max_tokens': 5,
        'n': 1,
        'stop': '\n'
    }
    response = requests.post(url, json=data, headers=headers)
    result = response.json()
    return int(result['choices'][0]['text'].strip())

@property
def recency(self):
    return self._recency

@recency.setter
def recency():
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
    self._recency = max(0, 1 - (recency_days / 365))  # Supone que después de un año, el evento no tiene relevancia

@property
def relevance(self):
    return self._relevance

@relevance.setter
def relevance(documents, query):
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

def generate_reflections(memory_stream):
    """
    Genera reflexiones a partir de las observaciones en la memoria del agente.

    Args:
    memory_stream (list of dict): La memoria del agente que contiene eventos con sus descripciones, importancia, etc.

    Returns:
    list of str: Lista de reflexiones generadas basadas en las observaciones más importantes y recientes.
    """
    # Filtrar eventos recientes y significativos para la reflexión
    recent_important_events = [event for event in memory_stream if event['recency'] > 0.5 and event['importance'] > 5]

    # Generar reflexiones si hay eventos suficientes
    reflections = []
    if not recent_important_events:
        reflections.append("No significant recent events to reflect on.")
    else:
        for event in recent_important_events:
            # Reflexión basada en la importancia y la descripción del evento
            reflection = f"Reflecting on the event: '{event['description']}', " \
                         f"which was significant with an importance of {event['importance']} " \
                         f"and recency of {event['recency']:.2f}."
            reflections.append(reflection)

    return reflections

def generate_observations(memory_stream, agent_actions, current_date, query_reference):
    """
    Genera y almacena observaciones con cálculos de importancia, recencia y relevancia.

    Args:
    memory_stream (list): Lista que representa la memoria del agente donde se almacenan las observaciones.
    agent_actions (list of str): Lista de acciones posibles para el agente.
    current_date (datetime): La fecha actual, que simula el tiempo real en el que ocurren las acciones.
    query_reference (str): Una consulta de referencia para calcular la relevancia de cada acción observada.

    Returns:
    None: Las observaciones se añaden directamente a la memoria del agente.
    """
    vectorizer = TfidfVectorizer()
    descriptions = [event['description'] for event in memory_stream]

    # Simula la percepción de una cantidad aleatoria de eventos por día
    num_events_today = random.randint(1, len(agent_actions))
    observed_actions = random.sample(agent_actions, num_events_today)

    for action in observed_actions:
        event_description = f"Observed action: {action}"
        importance = random.randint(1, 10)  # Asigna una importancia aleatoria a cada observación
        timestamp = current_date + datetime.timedelta(minutes=random.randint(1, 1440))  # Asigna un tiempo aleatorio durante el día
        recency = 1.0  # La recencia es máxima porque el evento acaba de ocurrir

        # Calcula la relevancia con respecto a la query de referencia
        if descriptions:
            tfidf_matrix = vectorizer.fit_transform(descriptions + [event_description, query_reference])
            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-2]).flatten()
            relevance = np.mean(cosine_similarities) if cosine_similarities.size > 0 else 0
        else:
            relevance = 0.0  # Si no hay descripciones previas, la relevancia es 0

        # Crea el evento de observación con todos los parámetros calculados
        observation = {
            'description': event_description,
            'importance': importance,
            'recency': recency,
            'relevance': relevance,
            'timestamp': timestamp
        }
        memory_stream.append(observation)


# Simulación de eventos
def run_simulation(events):
    """
    Ejecuta una simulación que evalúa la importancia, recencia y relevancia de una serie de eventos.
    
    Args:
    events (list of dict): Lista de eventos, donde cada evento es un diccionario con 'description' y 'date'.
    
    Returns:
    dict: Diccionario con los resultados de la simulación para cada evento.
    """
    results = {}
    current_date = datetime.datetime.now()
    documents = [event['description'] for event in events]  # Lista de descripciones de eventos para relevancia
    
    for event in events:
        description = event['description']
        event_date = datetime.datetime.strptime(event['date'], '%Y-%m-%d')
        
        # Calcular importancia, recencia y relevancia
        importance = set_importancy(description)
        recency = set_recency(event_date)
        relevance = set_relevance(documents, description)
        
        # Guardar resultados
        results[description] = {
            'importance': importance,
            'recency': recency,
            'relevance': relevance.tolist()  # Convertir array a lista para imprimir o almacenar
        }
       
    agent_actions = ['Attending a meeting', 'Writing a report', 'Having lunch with colleagues', 'Receiving a project assignment']
    current_date = datetime.datetime.now()
    query_reference = "work-related activities"

    generate_observations(memory_stream, agent_actions, current_date, query_reference)
        
    return results

memory_stream_example = [
    {'description': 'Graduated from college', 'importance': 9, 'recency': 0.9, 'timestamp': datetime.datetime.now()},
    {'description': 'Started a new job', 'importance': 8, 'recency': 0.8, 'timestamp': datetime.datetime.now()},
    {'description': 'Moved to a new city', 'importance': 7, 'recency': 0.7, 'timestamp': datetime.datetime.now()},
    {'description': 'Attended a professional networking event', 'importance': 5, 'recency': 0.65, 'timestamp': datetime.datetime.now()}
]

# Ejecutar la simulación
simulation_results = run_simulation(events)
print(simulation_results)