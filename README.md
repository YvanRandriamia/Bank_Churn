# Bank Churn Prediction

Ce projet a pour objectif de prédire les clients d'une banque susceptibles de clôturer leur compte ("churn").  
Grâce à des techniques de Machine Learning, nous avons analysé les comportements clients afin d’identifier les profils à risque et proposer des actions correctives.

# 📁 Contenu du projet

1. Introduction & Objectif  
2. Chargement des données  
3. Analyse exploratoire des données (EDA)  
4. Préparation des données  
5. Feature Engineering  
6. Modélisation avec plusieurs algorithmes  
   - Régression Logistique  
   - K-Nearest Neighbors  
   - Arbre de Décision  
   - Random Forest  
7. Comparaison, Interprétation des résultats et Recommandations  
8. Application interactive avec Streamlit dans le fichier (app.py)

## 📊 Modèles testés

| Modèle                 | Accuracy | F1-score (classe "Attrited") |
|------------------------|----------|-------------------------------|
| Logistic Regression    | 0.82     | 0.59                          |
| K-Nearest Neighbors    | 0.88     | 0.53                          |
| Decision Tree          | 0.96     | 0.86                          |
| Random Forest (best)   | 0.97     | 0.90                          |

Le modèle Random Forest est celui qui offre les meilleures performances.

## 🧠 Variables les plus importantes (Random Forest)

| Variable                    | Importance |
|----------------------------|------------|
| Total_Trans_Amt            | 0.232      |
| Total_Trans_Ct             | 0.231      |
| Total_Revolving_Bal        | 0.133      |
| Total_Amt_Chng_Q4_Q1       | 0.095      |
| Total_Relationship_Count   | 0.053      |

## 🛠️ Outils et bibliothèques utilisées

- Python 3.x  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  
- **Streamlit** (pour l’interface interactive)

## 🚀 Lancement de l’application Streamlit

Pour faciliter l’utilisation et permettre à tout utilisateur de tester la prédiction, une application interactive a été développée avec **Streamlit**.

### Comment lancer l’application :

1. Assurez-vous d’avoir installé toutes les dépendances :  
   ```bash
   pip install -r requirements.txt

2. ''bash 
    streamlit run app.py