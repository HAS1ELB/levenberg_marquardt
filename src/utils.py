import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(model_func, params, t, noise_std=0.1):
    """
    Génère des données synthétiques en ajoutant un bruit gaussien au modèle.

    Parameters:
        model_func: Fonction modèle utilisée pour générer les données (ex : func).
        params: Liste des paramètres réels du modèle.
        t: Données indépendantes (ex : temps).
        noise_std: Écart-type du bruit gaussien.

    Returns:
        np.ndarray: Données synthétiques bruitées.
    """
    y_clean = model_func(params, t)
    noise = np.random.normal(0, noise_std, len(t))
    return y_clean + noise


def plot_results(t, y_data, y_true, y_pred, title="Ajustement des données"):
    """
    Trace les données brutes, le modèle réel, et le modèle ajusté.

    Parameters:
        t: Données indépendantes (ex : temps).
        y_data: Données observées avec bruit.
        y_true: Valeurs générées par le modèle réel.
        y_pred: Valeurs générées par le modèle ajusté.
        title: Titre du graphique.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(t, y_data, label="Données avec bruit", color="red", s=10)
    plt.plot(t, y_true, label="Modèle réel", color="green", linewidth=2)
    plt.plot(t, y_pred, label="Modèle ajusté", color="blue", linestyle="--")
    plt.xlabel("Temps (t)")
    plt.ylabel("Valeurs (y)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def calculate_residuals(y_data, y_pred):
    """
    Calcule les résidus entre les données observées et prédites.

    Parameters:
        y_data: Données observées.
        y_pred: Données prédites par le modèle.

    Returns:
        np.ndarray: Résidus calculés (y_data - y_pred).
    """
    return y_data - y_pred


def mean_squared_error(y_data, y_pred):
    """
    Calcule l'erreur quadratique moyenne (MSE) entre les données observées et prédites.

    Parameters:
        y_data: Données observées.
        y_pred: Données prédites par le modèle.

    Returns:
        float: Valeur de l'erreur quadratique moyenne.
    """
    return np.mean((y_data - y_pred) ** 2)


def print_summary(true_params, optimized_params, mse):
    """
    Affiche un résumé des résultats de l'optimisation.

    Parameters:
        true_params: Paramètres réels utilisés pour générer les données synthétiques.
        optimized_params: Paramètres obtenus après l'optimisation.
        mse: Erreur quadratique moyenne entre les données réelles et ajustées.

    Returns:
        None
    """
    print("\n--- Résumé des résultats ---")
    print("Paramètres réels :       ", true_params)
    print("Paramètres optimisés :   ", optimized_params)
    print("Erreur quadratique moyenne (MSE) : {:.6f}".format(mse))
