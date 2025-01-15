from levenberg_marquardt import levenberg_marquardt
from utils import (
    generate_synthetic_data,
    plot_results,
    mean_squared_error,
    print_summary
)
import numpy as np


# Définition de la fonction modèle
def func(x, t):
    """
    Modèle de la fonction à ajuster : y = x[0] * exp(-x[1] * t)

    Parameters:
        x: Liste des paramètres [a, b].
        t: Données indépendantes (ex: temps).

    Returns:
        np.ndarray: Valeurs calculées par le modèle.
    """
    return x[0] * np.exp(-x[1] * t)


# Définition de la matrice Jacobienne
def jacobian(x, t):
    """
    Calcul de la matrice Jacobienne de la fonction modèle.

    Parameters:
        x: Liste des paramètres [a, b].
        t: Données indépendantes (ex: temps).

    Returns:
        np.ndarray: Matrice Jacobienne.
    """
    J = np.zeros((len(t), len(x)))
    J[:, 0] = np.exp(-x[1] * t)  # Dérivée partielle par rapport à x[0]
    J[:, 1] = -x[0] * t * np.exp(-x[1] * t)  # Dérivée partielle par rapport à x[1]
    return J


# Point d'entrée du script
if __name__ == "__main__":
    # Génération des données synthétiques
    t = np.linspace(0, 10, 100)  # Points dans le temps (100 points de 0 à 10)
    true_params = [2.5, 1.3]  # Paramètres réels du modèle
    y_data = generate_synthetic_data(func, true_params, t, noise_std=0.1)

    # Paramètres initiaux pour l'algorithme
    initial_guess = [1.0, 1.0]  # Devinez les paramètres initiaux

    # Exécution de la méthode de Levenberg-Marquardt
    optimized_params = levenberg_marquardt(func, jacobian, initial_guess, t, y_data)

    # Calcul de l'erreur quadratique moyenne
    mse = mean_squared_error(y_data, func(optimized_params, t))

    # Résumé des résultats
    print_summary(true_params, optimized_params, mse)

    # Comparaison des résultats avec le modèle optimisé
    plot_results(t, y_data, func(true_params, t), func(optimized_params, t))
