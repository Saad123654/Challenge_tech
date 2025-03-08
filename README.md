# Classification des Intentions d'un Chatbot d'Assistance Touristique
## Objectif du projet
Le but de ce projet est de proposer un algorithme permettant, étant donné un verbatim en entrée, de classifier l'intention exprimée. Les labels sont (translate, travel_alert, flight_status, lost_luggage, travel_suggestion, carry_on, book_hotel, book_flight, out_of_scope)

L'intention lost_luggage est particulière car elle redirige l'utilisateur vers un service client avec un coût élevé pour l'agence. Cette intention nécessite un traitement spécifique.

Les performances de nos modèles sont évaluées à l'aide des métriques suivantes :

- Accuracy : La précision générale du modèle, c'est-à-dire le pourcentage de prédictions correctes sur l'ensemble des données.
- Recall, Precision et la F1-score (La moyenne harmonique de la précision et du rappel )

Ces métriques sont calculées pour chaque classe individuellement, puis nous effectuons une moyenne pondérée ou une moyenne macro de ces valeurs.

## Installation des pré-requis
Exécutez la commande suivante pour installer toutes les bibliothèques nécessaires à partir du fichier requirements.txt
```sh
pip install -r requirements.txt

   ```

## Structure du projet

### Approche
Après avoir analysé les données et effectué un split stratifié dans le notebook `analyse_des_données.ipynb`, nous avons procédé à l'augmentation des données de manière indépendante pour éviter tout problème de data leakage. Deux méthodes ont été utilisées :

1. **Augmentation avec LLama 3.1** :
   - Le modèle a été importé et entraîné avec un contexte spécifique (voir `notebook/data_augmentation.ipynb`).
   - Exécuté sur une GPU de Kaggle.
   - Données stockées dans `data/augmented_raw/`.
   - Nettoyage post-augmentation dans `notebook/post_augmentation_cleaning.ipynb`, puis stockage des données nettoyées dans `data/augmented/`.

2. **Augmentation avec Deepseek** :
   - Données stockées dans `data/deepseek_augment/`.

### Entraînement des modèles

#### Modèles TF-IDF + SVM
Pour chaque méthode d’augmentation, nous avons entraîné un pipeline TF-IDF suivi d'un SVM avec recherche du nombre optimal de `max_features`.
- Entraînement et test dans :
  - `notebook/trainining/tfidf.ipynb`
  - `notebook/deepseek_augmentation/tf_idf_ds.ipynb`
- Export des modèles en `.pkl`.

#### Fine-tuning de DistilBERT
Nous avons fine-tuné un modèle `DistilBERT-multilingual-uncased` après modification de sa couche de sortie en 9 (nombre de labels). 
- Stratégie : `early stopping` pour éviter le surapprentissage.
- Entraînement et test dans :
  - `notebook/trainining/finetuning_distilbert.ipynb`
  - `notebook/deepseek_augmentation/finetuning_distilbert_ds.ipynb`
- Export des modèles en `.pkl`.

Les modèles doivent être enregistrés dans `models/`, mais en raison de leur taille, ils ne sont pas inclus dans le dépôt Git. Ils sont disponibles sur Google Drive : [Lien vers Google Drive](https://drive.google.com/drive/folders/1qUiHWMbolFD2nONCIa7yBEkxeWqHxGpA?usp=sharing).

### Inférence
Le dossier `inference/` contient les scripts Python pour inférer les modèles. Pour effectuer une prédiction :
1. Placer le fichier à prédire dans `data/Inference/` et le renommer en `inference_data.csv`.
2. Exécuter la commande correspondante dans le dossier `inference/` :
   ```sh
   python distilbert.py
   python distilbert_ds.py
   python tfidf_svm.py
   python tfidf_svm_ds.py
   ```

### Résultats

1. TF-IDF + SVM avec données augmentées avec LLama 3.1
Classe	|Précision	Rappel	F1-score	Support
0	0.54	0.96	0.69	51
1	1.00	0.89	0.94	9
2	0.86	0.90	0.88	21
3	0.64	0.54	0.58	13
4	0.77	0.56	0.65	18
5	0.87	0.33	0.48	39
6	1.00	0.79	0.89	39
7	1.00	0.73	0.84	33
8	0.61	0.85	0.71	20
Moyenne	0.81	0.73	0.74	243

3. TF-IDF + SVM avec données augmentées avec Deepseek
Classe	Précision	Rappel	F1-score	Support
0	0.73	1.00	0.85	30
1	1.00	0.90	0.95	10
2	0.83	1.00	0.91	10
3	1.00	0.80	0.89	10
4	1.00	0.80	0.89	10
5	1.00	0.50	0.67	10
6	1.00	1.00	1.00	10
7	1.00	0.80	0.89	10
8	1.00	0.90	0.95	10
Moyenne	0.95	0.86	0.89	110

5. DistilBERT avec données augmentées avec LLama 3.1
Classe	Précision	Rappel	F1-score	Support
out_of_scope	0.78	0.69	0.73	51
translate	0.89	0.89	0.89	9
book_flight	0.80	0.95	0.87	21
lost_luggage	0.69	0.69	0.69	13
travel_alert	1.00	0.39	0.56	18
travel_suggestion	0.60	0.87	0.71	39
carry_on	1.00	1.00	1.00	39
book_hotel	1.00	0.88	0.94	33
flight_status	0.89	0.85	0.87	20
Moyenne	0.85	0.80	0.81	243

6. DistilBERT avec données augmentées avec Deepseek
Classe	Précision	Rappel	F1-score	Support
out_of_scope	0.97	1.00	0.98	30
translate	1.00	1.00	1.00	10
book_flight	1.00	1.00	1.00	10
lost_luggage	1.00	0.90	0.95	10
travel_alert	0.90	0.90	0.90	10
travel_suggestion	0.89	0.80	0.84	10
carry_on	0.91	1.00	0.95	10
book_hotel	1.00	1.00	1.00	10
flight_status	1.00	1.00	1.00	10
Moyenne	0.96	0.96	0.96	110

| Modèle | Precision | Recall | F1-score | Accuracy |
|--------|-----------|--------|----------|----------|
| **TF-IDF + SVM (LLama)** | 0.81 | 0.73 | 0.74 | 0.73 |
| **TF-IDF + SVM (Deepseek)** | 0.81 | 0.73 | 0.74 | 0.73 |
| **DistilBERT (LLama)** | 0.85 | 0.80 | 0.81 | 0.81 |
| **DistilBERT (Deepseek)** | 0.96 | 0.96 | 0.96 | 0.96 |

Les performances montrent que le fine-tuning de `DistilBERT` avec l’augmentation `Deepseek` offre les meilleurs résultats avec une **accuracy de 96%**.

