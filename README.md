# IHSAS - Analyse des Sentiments en Arabe

## Nom de l'application : IHSAS

### Code d'Analyse des Sentiments en Arabe

#### Backend (`backend.py`)

1. **Installation des bibliothèques :**
Utilisez les commandes pip pour installer les bibliothèques nécessaires, telles que **aaransia**, **nltk**, **tashaphyne**, **arabic-reshaper**, et **wordcloud**.
```bash
pip install aaransia
```
```bash
pip install nltk
```
```bash
pip install tashaphyne
```
```bash
pip install --upgrade arabic-reshaper
pip install wordcloud
```
```bash
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

2. **Chargement des données :**
- Données d'entraînement et de test à partir de fichiers CSV.

3. **Prétraitement des données :**
- Translittération, normalisation, nettoyage, suppression des stopwords.

4. **Entraînement du modèle :**
- Modèle de Naive Bayes multinomial pour l'analyse des sentiments.

5. **API Flask :**
- Points d'extrémité pour la classification de texte et le téléchargement de fichiers CSV.

#### Interface Utilisateur (`design.css`, `home.html`, `script.js`)

1. **Page d'accueil :**
- Présentation du projet et de l'équipe.

2. **Analyse des Sentiments de Phrases :**
- Saisie d'une phrase en arabe pour une analyse immédiate.

3. **Téléchargement de Fichier :**
- Téléchargement d'un fichier texte (.txt) pour une analyse en masse.

4. **Styles CSS :**
- Mise en page de la page web.

5. **Script JavaScript :**
- Envoi de requêtes au backend et manipulation des résultats.

### Dataset

- Utilisation du jeu de données [ArSarcasm-v2](https://www.aclweb.org/anthology/2020.osact-1.5/).

## Exécution du Projet

1. **Exécution du Serveur Flask :**
```bash
python backend.py
```


3. **Utilisation de l'Interface Utilisateur :**
- Ouvrez `home.html` dans un navigateur.

4. **Classification de Texte :**
- Saisissez un texte en arabe et cliquez sur "Classify" pour obtenir le résultat.

5. **Téléchargement de Fichier :**
- Sélectionnez un fichier texte (.txt) et cliquez sur "Upload File" pour analyser et télécharger le résultat en CSV.

## Analyse des Sentiments en Arabe sur Kaggle

Vous pouvez retrouver le code complet de l'analyse des sentiments en arabe sur mon profile. Voici le lien vers le code source :

[Analyse des Sentiments en Arabe sur Kaggle](https://www.kaggle.com/code/ouaraskhelilrafik/analyse-sentiments-en-arabe)


## Auteur

- OUARAS Khelil Rafik

## Citation du Dataset

Si vous utilisez le dataset ArSarcasm-v2 dans votre projet de recherche, veuillez citer l'article original et le dataset de la manière suivante :

```bibtex
@inproceedings{abufarha-etal-2021-arsarcasm-v2,
  title = "Overview of the WANLP 2021 Shared Task on Sarcasm and Sentiment Detection in Arabic",
  author = "Abu Farha, Ibrahim and Zaghouani, Wajdi and Magdy, Walid",
  booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
  month = "April",
  year = "2021",
}
