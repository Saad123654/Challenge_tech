import pandas as pd
import pickle
import os 
import torch

# Chemin vers le répertoire contenant le modèle sauvegardé
pth = os.path.abspath(os.path.join(os.getcwd(), ".."))
MODEL_DIR = os.path.abspath(os.path.join(pth, "models/DistilBERT_finetuned.pkl"))
TOK_DIR = os.path.abspath(os.path.join(pth, "models/tokenizer.pkl"))

# Charger le tokenizer sauvegardé
with open(TOK_DIR, 'rb') as file:
    tokenizer = pickle.load(file)
# Charger le modèle sauvegardé
with open(MODEL_DIR, 'rb') as file:
    model = pickle.load(file)
print("Modèle chargé avec succès.")

# Charger le fichier CSV contenant les données à prédire
DATA_DIR = os.path.abspath(os.path.join(pth, "data/inference/inference_data.csv"))
df = pd.read_csv(DATA_DIR)


# Fonction de tokenisation
def tokenize_with_progress(data, desc):
    encodings = tokenizer(
        list(data),  # Convertir la colonne pandas en liste
        truncation=True,
        padding="max_length",  # Forcer la même longueur
        max_length=512,
        return_tensors="pt"  # Retourner directement un tenseur PyTorch
    )
    return {key: encodings[key] for key in encodings}


# Fonction d'inférence qui réutilise le prétraitement
def predict_with_preprocessing(texts):
    # Tokeniser les textes
    tokenized_data = tokenize_with_progress(texts,"Tokenisation inference")
    
    # Passer les données tokenisées dans le modèle
    model.eval()  # Mettre le modèle en mode évaluation
    with torch.no_grad():
        outputs = model(**tokenized_data)
        logits = outputs.logits  # Logits de la dernière couche du modèle

    # Prédire les labels en utilisant la classe avec la probabilité la plus élevée
    predicted_labels = torch.argmax(logits, dim=1).numpy()

    # Convertir les indices des prédictions en labels
    #predicted_labels = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in predicted_labels]

    return predicted_labels


# Appel de la fonction de prédiction
df['predicted_label'] = predict_with_preprocessing(df['text'])

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


# Mapper les prédictions numériques vers les labels correspondants sous forme de chaînes de caractères
df['predictions'] = df['predicted_label'].map(inverse_label_mapping)
df = df.drop(columns=['predicted_label'])
# Sauvegarder les résultats dans un nouveau fichier CSV
RESULTAT_DIR= os.path.abspath(os.path.join(pth, "data/inference/inference_results_DisTilBERT.csv"))
df.to_csv(RESULTAT_DIR, index=False)

# Afficher un aperçu des résultats
print("Inférence terminée. Voici un aperçu des résultats :")
print(df.head())

