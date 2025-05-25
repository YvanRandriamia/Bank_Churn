import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Chargement du modèle, encodeur, et colonnes
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("encoder.pkl")
model_columns = joblib.load("columns_model.pkl")

st.title("Prédiction de départ client - Bank Churn")

st.markdown("""
Bienvenue dans cette application de prédiction de churn client bancaire.  
Veuillez renseigner les informations du client ci-dessous, puis cliquez sur **Prédire**.
""")

# Formulaire pour les inputs utilisateur avec aides (help=)
with st.form(key='churn_form'):
    Total_Trans_Amt = st.number_input("Total Trans Amount", min_value=0, help="Montant total des transactions.")
    Total_Trans_Ct = st.number_input("Total Trans Count", min_value=0, help="Nombre total de transactions.")
    Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amt Chng Q4 Q1", min_value=0.0, format="%.3f",
                                          help="Changement du montant total entre le trimestre 4 et 1.")
    Total_Revolving_Bal = st.number_input("Total Revolving Balance", min_value=0, help="Solde revolving total.")
    Credit_Limit = st.number_input("Credit Limit", min_value=0.0, help="Limite de crédit autorisée.")
    Customer_Age = st.number_input("Customer Age", min_value=18, help="Âge du client.")
    Months_on_book = st.number_input("Months on Book", min_value=0, help="Nombre de mois depuis l'ouverture du compte.")
    Avg_Open_To_Buy = st.number_input("Avg Open To Buy", min_value=0.0, help="Montant moyen disponible à dépenser.")
    Total_Relationship_Count = st.number_input("Total Relationship Count", min_value=0,
                                               help="Nombre total de produits possédés par le client.")
    Months_Inactive_12_mon = st.number_input("Months Inactive 12 Months", min_value=0,
                                             help="Nombre de mois inactifs sur les 12 derniers mois.")

    Gender = st.selectbox("Gender", options=["F", "M"], help="Genre du client.")
    Income_Category = st.selectbox("Income Category", options=[
        "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "abc"
    ], help="Catégorie de revenu.")
    Marital_Status = st.selectbox("Marital Status", options=["Single", "Married"], help="Statut marital.")
    Education_Level = st.selectbox("Education Level", options=[
        "Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate"
    ], help="Niveau d'éducation.")

    submit_button = st.form_submit_button(label='Prédire')

if submit_button:
    # Création DataFrame input numérique
    input_num = pd.DataFrame({
        'Total_Trans_Amt': [Total_Trans_Amt],
        'Total_Trans_Ct': [Total_Trans_Ct],
        'Total_Amt_Chng_Q4_Q1': [Total_Amt_Chng_Q4_Q1],
        'Total_Revolving_Bal': [Total_Revolving_Bal],
        'Credit_Limit': [Credit_Limit],
        'Customer_Age': [Customer_Age],
        'Months_on_book': [Months_on_book],
        'Avg_Open_To_Buy': [Avg_Open_To_Buy],
        'Total_Relationship_Count': [Total_Relationship_Count],
        'Months_Inactive_12_mon': [Months_Inactive_12_mon]
    })

    # Préparation données catégorielles sous forme DataFrame
    input_cat = pd.DataFrame({
        'Gender': [Gender],
        'Income_Category': [Income_Category],
        'Marital_Status': [Marital_Status],
        'Education_Level': [Education_Level]
    })

    # Encodage OneHot
    input_cat_encoded_array = encoder.transform(input_cat)
    input_cat_encoded_df = pd.DataFrame(input_cat_encoded_array, columns=encoder.get_feature_names_out(), index=input_cat.index)

    # Concaténation données numériques + encodées
    input_final = pd.concat([input_num, input_cat_encoded_df], axis=1)

    # Réordonner colonnes selon le modèle et remplir les absentes par 0
    input_final = input_final.reindex(columns=model_columns, fill_value=0)

    # Prédiction
    prediction = model.predict(input_final)[0]
    proba = model.predict_proba(input_final)[0][1]

    # Affichage résultat
    if prediction == 1:
        st.error(f"⚠️ Attention : Ce client a un risque élevé de quitter la banque. (Probabilité : {proba:.2f})")
    else:
        st.success(f"✅ Ce client a peu de risque de quitter la banque. (Probabilité : {proba:.2f})")

    # Stocker la probabilité dans une session pour résumé global
    if "probas" not in st.session_state:
        st.session_state.probas = []
    st.session_state.probas.append(proba)

# Résumé global avec graphique simple si au moins 1 proba
if "probas" in st.session_state and len(st.session_state.probas) > 0:
    st.markdown("### Résumé des probabilités de churn prédites jusqu'à présent")
    df_probas = pd.DataFrame(st.session_state.probas, columns=["Probabilité de churn"])
    fig, ax = plt.subplots()
    ax.hist(df_probas["Probabilité de churn"], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel("Probabilité de churn")
    ax.set_ylabel("Nombre de clients")
    st.pyplot(fig)
