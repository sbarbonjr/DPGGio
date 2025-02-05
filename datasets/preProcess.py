import pandas as pd
import os
import argparse

def remove_categorical_and_binary_features(df):
    # Identifica colonne categoriche e binarie
    columns_to_remove = []
    for col in df.columns:
        if df[col].dtype == 'object' or (df[col].nunique() == 2 and set(df[col].unique()) <= {-1, 0, 1}):
            columns_to_remove.append(col)

    # Rimuove le colonne categoriche e binarie
    df_cleaned = df.drop(columns=columns_to_remove)
    
    return df_cleaned

def preprocess_and_save_csv(input_file, column_to_remove=None):
    # Legge il dataset
    df = pd.read_csv(input_file)
    
    # Sostituisce gli spazi nei nomi delle colonne con "_"
    df.columns = df.columns.str.replace(' ', '_')
    print(f"Nomi delle colonne modificati per rimuovere gli spazi.")
    
    # Rimuove i campioni con valori NaN o vuoti
    df_cleaned = df.dropna()
    print(f"Rimosse {len(df) - len(df_cleaned)} righe contenenti valori NaN o vuoti.")
    
    # Preprocessa il dataset rimuovendo colonne categoriche e binarie
    df_cleaned = remove_categorical_and_binary_features(df_cleaned)
    
    # Converte tutti i valori numerici (tranne le colonne con nomi non numerici) in float
    for col in df_cleaned.columns:
        # Se la colonna contiene valori numerici (float, int) o può essere convertita
        if df_cleaned[col].dtype != 'object':
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Rimuove la colonna specificata dall'utente (se presente)
    if column_to_remove and column_to_remove in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=[column_to_remove])
        print(f"Colonna '{column_to_remove}' rimossa dal dataset.")
    elif column_to_remove:
        print(f"Attenzione: la colonna '{column_to_remove}' non esiste nel dataset.")
    
    # Crea il nome del file di output aggiungendo "_preprocessed"
    file_name, file_extension = os.path.splitext(input_file)
    output_file = f"{file_name}_preprocessed{file_extension}"
    
    # Salva il nuovo dataset in un file CSV
    df_cleaned.to_csv(output_file, index=False)
    print(f"Dataset preprocessato salvato come: {output_file}")

if __name__ == "__main__":
    # Definisci il parser degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Preprocessa un dataset rimuovendo feature categoriche, binarie, campioni con NaN, e una colonna specificata.")
    
    # Argomento per il file di input
    parser.add_argument("--input_file", type=str, help="Il file CSV da preprocessare")
    
    # Argomento opzionale per il nome della colonna da rimuovere
    parser.add_argument("--remove_column", type=str, help="Nome della colonna da rimuovere", default=None)
    
    # Parsing degli argomenti
    args = parser.parse_args()
    
    # Chiama la funzione per preprocessare il CSV e salvarlo
    preprocess_and_save_csv(args.input_file, args.remove_column)
