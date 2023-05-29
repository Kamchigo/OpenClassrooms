import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Chargement des données d'entraînement depuis votre source de données ou votre fichier
# CSV d'entraînement
# Assurez-vous de remplacer "chemin_vers_votre_fichier_csv_d'entrainement.csv" par le
# chemin d'accès à votre fichier CSV d'entraînement
df_train = pd.read_csv("dfbilletsog.csv")

# Variable dépendante
y_train_log = df_train["is_genuine"]

# Variables indépendantes
X_train_log = df_train[["diagonal", "height_left",
                        "height_right", "margin_low", "margin_up", "length"]]

# Création d'une instance de StandardScaler
scaler = StandardScaler()

# Ajustement du StandardScaler sur les données d'entraînement
scaler.fit(X_train_log)

# Création d'une instance de modèle
model = LogisticRegression()

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train_log, y_train_log)

# Fonction de prédiction


def predict_billets(df_test):
    # Exclusion de la colonne 'id'
    df_test_features = df_test.drop('id', axis=1)

    # Transformation des données de test à l'aide du StandardScaler ajusté
    df_test_scaled = scaler.transform(df_test_features)

    # Prédiction sur les données de test
    y_pred_proba = model.predict_proba(df_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.4).astype(int)

    # Ajouter les prédictions au DataFrame de test
    df_test["is_genuine_pred"] = y_pred

    # Remplacer les valeurs 0 et 1 par "FAUX" et "VRAI"
    df_test["is_genuine_pred"] = df_test["is_genuine_pred"].map(
        {0: "FAUX", 1: "VRAI"})

    return df_test

# Configuration de l'application Streamlit


def main():
    st.title("Application de détection de faux billets")
    st.write("Cette application a pour fonction de prédire l'authenticité d'un jeu de "
             "billets en fonction des caractéristiques de chacun des billets."
             "Elle repose sur un algorithme de régression logistique binaire.")

    # Téléchargement du fichier CSV de test
    uploaded_file = st.file_uploader("Uploader le fichier CSV", type="csv")

    if uploaded_file is not None:
        # Lecture du fichier CSV téléchargé
        df_test = pd.read_csv(uploaded_file)

        # Prédiction sur les données de test
        df_result = predict_billets(df_test)

        # Affichage des résultats de prédiction
        st.write("Résultats de la prédiction :")

        # Filtre pour sélectionner les vrais ou les faux billets
        filter_type = st.selectbox("Sélectionner le type de billets :", [
                                   "Tous", "VRAI", "FAUX"])

        if filter_type == "Tous":
            # Afficher tous les billets
            st.write(df_result)
        else:
            # Filtrer les billets en fonction du type sélectionné
            df_filtered = df_result[df_result["is_genuine_pred"]
                                    == filter_type]
            st.write("Billets", filter_type, " :")
            st.write(df_filtered)

        # Diagramme en camembert pour la répartition des vrais et faux billets
        fig, ax = plt.subplots()
        labels = df_result["is_genuine_pred"].value_counts().index
        sizes = df_result["is_genuine_pred"].value_counts().values
        colors = ["green", "red"]
        explode = (0.1, 0)  # Explosion de la première part du camembert

        ax.pie(sizes, explode=explode, labels=labels,
               colors=colors, autopct='%1.1f%%', startangle=90)
        # Aspect ratio égal pour s'assurer que le camembert est circulaire
        ax.axis('equal')

        # Légende adaptée
        legend_labels = ['VRAI', 'FAUX']
        legend_colors = ['green', 'red']
        legend_texts = [f"{label} ({size})" for label,
                        size in zip(labels, sizes)]

        ax.legend(legend_labels, title="Type de billets", loc="best", bbox_to_anchor=(
            1, 0.5), labels=legend_texts, facecolor='white', edgecolor='black')
        # Ajout du titre
        ax.set_title("Répartition des billets par authenticité")
        # Affichage du diagramme en camembert
        st.pyplot(fig)


if __name__ == "__main__":
    main()
