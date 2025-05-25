import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des objets
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("encoder.pkl")
model_columns = joblib.load("columns_model.pkl")

# Configuration du style
st.set_page_config(page_title="Bank Churn Prediction", layout="wide")

# Titre principal
st.markdown("<h1 style='color:#c084fc;'>💼 Bank Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("Prédisez si un client risque de quitter la banque.")

# Formulaire d'entrée utilisateur
with st.form("formulaire"):
    st.subheader("📋 Informations client")
    col1, col2 = st.columns(2)

    with col1:
        Total_Trans_Amt = st.number_input("Total Trans Amount", min_value=0)
        Total_Trans_Ct = st.number_input("Total Trans Count", min_value=0)
        Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amt Chng Q4 Q1", min_value=0.0, format="%.3f")
        Total_Revolving_Bal = st.number_input("Total Revolving Balance", min_value=0)
        Credit_Limit = st.number_input("Credit Limit", min_value=0.0)

    with col2:
        Customer_Age = st.number_input("Customer Age", min_value=18)
        Months_on_book = st.number_input("Months on Book", min_value=0)
        Avg_Open_To_Buy = st.number_input("Avg Open To Buy", min_value=0.0)
        Total_Relationship_Count = st.number_input("Total Relationship Count", min_value=0)
        Months_Inactive_12_mon = st.number_input("Months Inactive 12 Months", min_value=0)

    Gender = st.selectbox("Gender", ["F", "M"])
    Income_Category = st.selectbox("Income Category", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "abc"])
    Marital_Status = st.selectbox("Marital Status", ["Single", "Married"])
    Education_Level = st.selectbox("Education Level", ["Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate"])

    submit_button = st.form_submit_button("🎯 Prédire")

# Traitement
if submit_button:
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

    input_cat = pd.DataFrame({
        'Gender': [Gender],
        'Income_Category': [Income_Category],
        'Marital_Status': [Marital_Status],
        'Education_Level': [Education_Level]
    })

    input_cat_encoded = pd.DataFrame(
        encoder.transform(input_cat),
        columns=encoder.get_feature_names_out()
    )

    input_final = pd.concat([input_num, input_cat_encoded], axis=1).reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_final)[0]
    proba = model.predict_proba(input_final)[0][1]

    # Résumé du client
    with st.container():
        st.markdown("---")
        st.markdown("### 🧾 Résumé du client")
        st.dataframe(input_num.style.set_properties(**{'border-radius': '10px'}), use_container_width=True)

        st.markdown("### 🔍 Résultat de la prédiction")
        if prediction == 1:
            st.error(f"⚠️ Ce client a un **risque ÉLEVÉ** de quitter la banque. (Probabilité : {proba:.2f})")
        else:
            st.success(f"✅ Ce client a **peu de risque** de partir. (Probabilité : {proba:.2f})")

    # Sauvegarder l'historique
    client_info = input_num.copy()
    client_info["Probabilité de churn"] = proba
    client_info["Risque élevé"] = "Oui" if prediction == 1 else "Non"

    if "historique_clients" not in st.session_state:
        st.session_state.historique_clients = pd.DataFrame()

    st.session_state.historique_clients = pd.concat(
        [st.session_state.historique_clients, client_info],
        ignore_index=True
    )

# Affichage historique
if "historique_clients" in st.session_state and not st.session_state.historique_clients.empty:
    st.markdown("### 🧠 Historique des prédictions")
    df = st.session_state.historique_clients.copy()
    df["Index"] = df.index

    st.dataframe(df.style.set_properties(**{'border-radius': '10px'}), use_container_width=True)

    to_delete = st.multiselect("Sélectionnez les lignes à supprimer :", df["Index"])

    if st.button("🗑️ Supprimer les lignes sélectionnées"):
        st.session_state.historique_clients.drop(index=to_delete, inplace=True)
        st.session_state.historique_clients.reset_index(drop=True, inplace=True)
        st.success("Lignes supprimées.")

