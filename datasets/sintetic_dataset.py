import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
from sklearn.ensemble import IsolationForest


# === 1. Generazione degli inliers ===
# Genera 100 campioni inliers a 6 features attorno a un unico centro (deviazione standard = 1.0)
X_inliers, _ = make_blobs(n_samples=200, centers=1, n_features=6, cluster_std=1.0, random_state=42)

# Calcola la media e la deviazione standard per ogni feature degli inliers
mean_inliers = np.mean(X_inliers, axis=0)
std_inliers = np.std(X_inliers, axis=0)

# === 2. Selezione e modifica degli outliers ===
rng = np.random.RandomState(42)
n_outliers = 4
indices_outliers = rng.choice(np.arange(X_inliers.shape[0]), size=n_outliers, replace=False)

# Crea una copia dei campioni selezionati da modificare
X_outliers = X_inliers[indices_outliers].copy()

# Definizione dei gruppi:
# - Gruppo 1 (primo outlier): 2 features anomale, valore = mean ± 3*std
# - Gruppo 2 (i successivi 2 outliers): 4 features anomale, valore = mean ± 2*std
# - Gruppo 3 (ultimo outlier): 1 feature anomala, valore = mean ± 2*std
for i in range(n_outliers):
    if i < 1:
        num_features_anomaly = 3
        factor = 5
    elif i < 3:
        num_features_anomaly = 4
        factor = 5
    else:
        num_features_anomaly = 2
        factor = 5

    # Seleziona casualmente (senza ripetizioni) le feature da modificare
    features_to_modify = rng.choice(np.arange(6), size=num_features_anomaly, replace=False)
    for feat in features_to_modify:
        # Per ogni feature, scegli se sommare o sottrarre il valore anomalo
        sign = rng.choice([-1, 1])
        # Imposta il nuovo valore: mean ± factor*std
        X_outliers[i, feat] = mean_inliers[feat] + sign * factor * std_inliers[feat]

# Sostituisci i campioni originali con quelli modificati (outliers)
X_final = X_inliers.copy()
X_final[indices_outliers] = X_outliers

# Crea l'array delle etichette ground truth: 1 per gli inliers, -1 per gli outliers
y_true = np.ones(X_final.shape[0])
y_true[indices_outliers] = -1

print("Campioni anomali (modificati):")
print(X_final[indices_outliers])


# === 3. Creazione e salvataggio del DataFrame ===
df = pd.DataFrame(X_final, columns=[f"Feature{i}" for i in range(6)])
df["Edited"] = y_true

df.to_csv("sintetic.csv", index=False)
print("Il dataset è stato salvato in 'sintetic.csv'.")


# === 4. Applicazione di Isolation Forest per rilevare le anomalie ===
features = [f"Feature{i}" for i in range(6)]
clf = IsolationForest(n_estimators=200, random_state=42, contamination='auto')
clf.fit(df[features])
pred = clf.predict(df[features])

# Calcola gli anomaly score e aggiungili al DataFrame
df["ScoreSample"] = clf.score_samples(df[features])

# Calcola gli anomaly score e aggiungili al DataFrame
df["AnomalyScore(paper)"] = clf.score_samples(df[features])*(-1)

# Calcola gli anomaly score e aggiungili al DataFrame
df["AnomalyScore(scikit)"] = clf.decision_function(df[features])

# Aggiungi le predizioni come nuova colonna (1 = inlier, -1 = outlier)
df["iFLabel"] = pred
anomalies = df[df["iFLabel"] == -1]
print("\n", anomalies)

# Stampa l'intero dataset con tutti gli anomaly score
print("\nDataset completo con AnomalyScore:")
print(df)

# Assegna un ID progressivo agli outliers rilevati da Isolation Forest
df.loc[df['iFLabel'] == -1, 'iFAnomalyID'] = range(0, df[df['iFLabel'] == -1].shape[0])
df.loc[df['iFLabel'] == 1, 'iFAnomalyID'] = np.nan

# === 5. Creazione del pair plot con annotazioni per le anomalie rilevate da Isolation Forest ===
g = sns.pairplot(
    df, 
    vars=features, 
    hue="iFLabel",
    palette={1: (47/255, 103/255, 177/255), -1: (191/255, 44/255, 35/255)},
    diag_kind="hist"
)
plt.suptitle("Pair Plot (Isolation Forest predictions)", y=1.02)

# Annotazione: in ogni subplot (esclusi quelli diagonali), per ogni outlier, scriviamo il suo IFAnomalyID
for i, row_var in enumerate(g.y_vars):
    for j, col_var in enumerate(g.x_vars):
        # Salta i grafici diagonali
        if i == j:
            continue
        ax = g.axes[i, j]
        # Seleziona solo i campioni rilevati come outlier da Isolation Forest
        outliers_df = df[df['iFLabel'] == -1]
        for idx, row in outliers_df.iterrows():
            x_val = row[col_var]
            y_val = row[row_var]
            anomaly_id = int(row['iFAnomalyID'])
            ax.annotate(
                f"[{anomaly_id}]",
                (x_val, y_val),
                xytext=(5, 0),
                textcoords="offset points",
                color="black",
                fontsize=8,
                ha="left",
                va="center"
            )

plt.show()

print("")
subprocess.run(["python", "preProcess.py", "--input_file", "sintetic.csv", "--remove_column", "Label"])