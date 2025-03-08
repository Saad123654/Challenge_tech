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
  - `notebook/deepseek_augmentation/tf_idf_ds/finetuning_distilbert_ds.ipynb`
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

| Modèle | Precision | Recall | F1-score | Accuracy |
|--------|-----------|--------|----------|----------|
| **TF-IDF + SVM (LLama)** | 0.81 | 0.73 | 0.74 | 0.73 |
| **TF-IDF + SVM (Deepseek)** | 0.81 | 0.73 | 0.74 | 0.73 |
| **DistilBERT (LLama)** | 0.85 | 0.80 | 0.81 | 0.81 |
| **DistilBERT (Deepseek)** | 0.96 | 0.96 | 0.96 | 0.96 |

Les performances montrent que le fine-tuning de `DistilBERT` avec l’augmentation `Deepseek` offre les meilleurs résultats avec une **accuracy de 96%**.

