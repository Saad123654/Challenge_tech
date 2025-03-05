import pandas as pd
import pickle
import os 

# Chemin vers le répertoire contenant le modèle sauvegardé
pth = os.path.abspath(os.path.join(os.getcwd(), ".."))
MODEL_DIR = os.path.abspath(os.path.join(pth, "models/svm_ds_model.pkl"))

# Charger le modèle sauvegardé
with open(MODEL_DIR, 'rb') as file:
    model_pipeline = pickle.load(file)
print("Modèle chargé avec succès.")
# Charger le fichier CSV contenant les données à prédire
DATA_DIR = os.path.abspath(os.path.join(pth, "data/inference/inference_data.csv"))
df = pd.read_csv(DATA_DIR)

# Vérifier que le fichier CSV contient bien une colonne 'text'
if 'text' not in df.columns:
    raise ValueError("Le fichier CSV doit contenir une colonne 'text' pour effectuer les prédictions.")
print("Données chargées avec succès.")
# Effectuer l'inférence en utilisant le modèle chargé
X_input = df['text']  
y_pred = model_pipeline.predict(X_input)
label_mapping={'out_of_scope': 0,
 'translate': 1,
 'book_flight': 2,
 'lost_luggage': 3,
 'travel_alert': 4,
 'travel_suggestion': 5,
 'carry_on': 6,
 'book_hotel': 7,
 'flight_status': 8}

# Inverser le dictionnaire pour avoir un mapping des indices vers les labels
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Ajouter les prédictions encodées au DataFrame
df['predictions_enc'] = y_pred

# Mapper les prédictions numériques vers les labels correspondants sous forme de chaînes de caractères
df['predictions'] = df['predictions_enc'].map(inverse_label_mapping)

# Sauvegarder les résultats dans un nouveau fichier CSV
RESULTAT_DIR= os.path.abspath(os.path.join(pth, "data/inference/inference_results_svm_ds.csv"))
df.to_csv(RESULTAT_DIR, index=False)

# Afficher un aperçu des résultats
print("Inférence terminée. Voici un aperçu des résultats :")
print(df.head())
