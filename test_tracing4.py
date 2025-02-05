import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


csv_file = '../Datasets/IrisShorted33_preprocessed.csv'  # Sostituisci con il tuo percorso del file CSV

df = pd.read_csv(csv_file)

iso_forest = IsolationForest(contamination=0.34, n_estimators=50, random_state=31, max_samples=11)
    
data = []
for index, row in df.iterrows():
    data.append([row[j] for j in df.columns])
data = np.array(data)
    
# Addestra il modello con i dati
iso_forest.fit(data)
iso_forest.predict(data)
    
# Funzione per ottenere i campioni per ogni foglia in un albero
def get_samples_in_leaves(tree, data):
    leaf_indices = tree.apply(data)  # Ottieni l'indice della foglia per ogni campione
    leaf_samples = {}  # Dizionario per raccogliere i campioni per ogni foglia
    
    for i, leaf in enumerate(leaf_indices):
        if leaf not in leaf_samples:
            leaf_samples[leaf] = []
        leaf_samples[leaf].append(i)  # Aggiungi l'indice del campione alla foglia corrispondente
    
    return leaf_samples

# Estrai i campioni per ogni foglia per ogni albero
for i, tree in enumerate(iso_forest.estimators_):
    print(f"Albero {i}:")
    leaf_samples = get_samples_in_leaves(tree, data)
    
    # Mostra i campioni per ogni foglia
    for leaf, samples in leaf_samples.items():
        print(f"  Foglia {leaf}: {len(samples)} campioni")
        print(f"    Indici dei campioni: {samples}")
    print("\n")
    
# Previsione delle anomalie (1 per normale, -1 per anomalo)
predictions = iso_forest.predict(data)

# Trova gli indici delle anomalie (campioni con previsione -1)
anomalies = np.where(predictions == -1)[0]

# Stampa gli indici delle anomalie
print("Indici delle anomalie trovate:")
print(anomalies)
