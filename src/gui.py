import sys
import os
# Ajouter le dossier racine du projet au chemin d'exécution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import streamlit as st
import numpy as np
from src.levenberg_marquardt import levenberg_marquardt
from src.utils import generate_synthetic_data, plot_results, mean_squared_error
from src.main import func, jacobian

# Titre de l'application
st.title("Interface Levenberg-Marquardt")

# Section : Entrée des paramètres
st.sidebar.header("Paramètres de l'algorithme")
model_type = st.sidebar.selectbox("Type de modèle", ["exponential", "polynomial", "sinusoidal"])
initial_guess = st.sidebar.text_input("Paramètres initiaux (séparés par des virgules, ex: 1.0, 1.0)", "1.0, 1.0")
max_iter = st.sidebar.number_input("Nombre maximal d'itérations", min_value=10, max_value=1000, value=100, step=10)
tol = st.sidebar.number_input("Tolérance", min_value=1e-10, max_value=1e-3, value=1e-6, format="%.1e")
lambda_init = st.sidebar.number_input("Facteur d'amortissement initial", min_value=1e-4, max_value=1.0, value=0.01, format="%.2f")

# Section : Données
st.sidebar.header("Données")
data_option = st.sidebar.radio("Source des données", ["Générer synthétiquement", "Télécharger un fichier"])
if data_option == "Générer synthétiquement":
    t = np.linspace(0, 10, 100)
    true_params = [2.5, 1.3] if model_type == "exponential" else [2.5, 1.3, 0.5]
    noise_std = st.sidebar.slider("Niveau de bruit", 0.0, 1.0, 0.1)
    y_data = generate_synthetic_data(func, true_params, t, noise_std)
else:
    uploaded_file = st.sidebar.file_uploader("Télécharger un fichier CSV (t, y)", type=["csv"])
    if uploaded_file:
        data = np.loadtxt(uploaded_file, delimiter=",")
        t, y_data = data[:, 0], data[:, 1]
    else:
        st.warning("Veuillez télécharger un fichier pour continuer.")
        st.stop()

# Bouton pour exécuter l'algorithme
if st.button("Exécuter l'algorithme"):
    # Conversion des paramètres initiaux
    try:
        initial_guess = [float(x) for x in initial_guess.split(",")]

        # Vérification du nombre de paramètres requis
        if model_type == "polynomial" and len(initial_guess) < 3:
            st.error("Pour le modèle polynomial, vous devez fournir au moins 3 paramètres initiaux.")
            st.stop()
        elif model_type == "sinusoidal" and len(initial_guess) < 3:
            st.error("Pour le modèle sinusoidal, vous devez fournir au moins 3 paramètres initiaux.")
            st.stop()
        elif model_type == "exponential" and len(initial_guess) < 2:
            st.error("Pour le modèle exponentiel, vous devez fournir au moins 2 paramètres initiaux.")
            st.stop()
    except ValueError:
        st.error("Veuillez entrer des paramètres initiaux valides.")
        st.stop()

    # Exécution de l'algorithme
    try:
        optimized_params = levenberg_marquardt(
            func, jacobian, initial_guess, t, y_data, max_iter, tol, lambda_init, model_type
        )
        mse = mean_squared_error(y_data, func(optimized_params, t, model_type))

        # Affichage des résultats
        st.subheader("Résultats de l'optimisation")
        st.write(f"Paramètres optimisés : {optimized_params}")
        st.write(f"Erreur quadratique moyenne (MSE) : {mse:.6f}")

        # Tracer et afficher les graphiques
        fig = plot_results(t, y_data, func(true_params, t, model_type), func(optimized_params, t, model_type))
        st.pyplot(fig)  # Affiche le graphique dans l'application Streamlit
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'exécution : {e}")

# Message si aucune donnée n'est chargée
if data_option == "Télécharger un fichier" and not uploaded_file:
    st.warning("Veuillez télécharger un fichier pour continuer.")
