import pandas as pd
from scipy.io import loadmat

name_dataset = "shuttle"

# Carica il file .mat
data = loadmat(f"{name_dataset}.mat")

# Visualizza tutte le chiavi del dizionario (variabili salvate nel file .mat)
print("Chiavi nel file .mat:", data.keys())

# Estrai le variabili X e y dal file .mat
X = data['X']  # Dataset principale
y = data['y']  # Etichette

# Crea un DataFrame da X
df_X = pd.DataFrame(X)

# Aggiungi la colonna 'y' al DataFrame con le etichette
df_X['y'] = y

# Genera nomi generici per le feature (f1, f2, ..., fn)
feature_names = [f'f{i+1}' for i in range(X.shape[1])]

# Assegna i nomi delle feature solo alle colonne di X
df_X.columns = feature_names + ['y']  # La colonna y è già stata aggiunta

# Visualizza il DataFrame
print(df_X)

# Conta il numero di campioni che hanno il valore 1 in 'y' (anomalie)
anomalia_count = df_X[df_X['y'] == 1].shape[0]
print(f"Numero di campioni che hanno il valore 1 come 'y' (anomalie): {anomalia_count}")

# Salva il DataFrame in un file CSV
df_X.to_csv(f'{name_dataset}.csv', index=False)
