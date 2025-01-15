import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.levenberg_marquardt import levenberg_marquardt
from src.utils import generate_synthetic_data
import matplotlib.pyplot as plt


# Fonction modèle et jacobienne pour les tests
def func(x, t, model_type="exponential"):
    """
    Modèle de la fonction à ajuster.
    """
    if model_type == "exponential":
        return x[0] * np.exp(-x[1] * t)
    elif model_type == "polynomial":
        return x[0] + x[1] * t + x[2] * t**2
    elif model_type == "sinusoidal":
        return x[0] * np.sin(x[1] * t + x[2])
    else:
        raise ValueError(f"Modèle {model_type} non supporté.")


def jacobian(x, t, model_type="exponential"):
    """
    Calcul de la matrice Jacobienne de la fonction modèle.
    """
    if model_type == "exponential":
        J = np.zeros((len(t), len(x)))
        J[:, 0] = np.exp(-x[1] * t)
        J[:, 1] = -x[0] * t * np.exp(-x[1] * t)
        return J
    elif model_type == "polynomial":
        J = np.ones((len(t), len(x)))
        J[:, 1] = t
        J[:, 2] = t**2
        return J
    elif model_type == "sinusoidal":
        J = np.zeros((len(t), len(x)))
        J[:, 0] = np.sin(x[1] * t + x[2])
        J[:, 1] = x[0] * t * np.cos(x[1] * t + x[2])
        J[:, 2] = x[0] * np.cos(x[1] * t + x[2])
        return J
    else:
        raise ValueError(f"Modèle {model_type} non supporté.")


class TestLevenbergMarquardt(unittest.TestCase):
    def test_levenberg_marquardt(self):
        # Générer des données synthétiques
        t = np.linspace(0, 10, 100)
        true_params = [2.5, 1.3]
        y_data = generate_synthetic_data(func, true_params, t, noise_std=0.1)

        # Paramètres initiaux pour l'algorithme
        initial_guess = [1.0, 1.0]

        # Appeler l'algorithme Levenberg-Marquardt
        optimized_params = levenberg_marquardt(func, jacobian, initial_guess, t, y_data)

        # Tester si les paramètres optimisés sont proches des vrais paramètres
        np.testing.assert_almost_equal(optimized_params[0], true_params[0], decimal=1)
        np.testing.assert_almost_equal(optimized_params[1], true_params[1], decimal=1)

    def test_convergence(self):
        # Générer des données synthétiques
        t = np.linspace(0, 10, 100)
        true_params = [3.0, 1.0]
        y_data = generate_synthetic_data(func, true_params, t, noise_std=0.1)

        # Paramètres initiaux pour l'algorithme
        initial_guess = [1.0, 0.5]

        # Appeler l'algorithme Levenberg-Marquardt
        optimized_params = levenberg_marquardt(func, jacobian, initial_guess, t, y_data)

        # Vérifier la convergence en comparant les paramètres optimisés avec les vrais paramètres
        self.assertTrue(np.allclose(optimized_params, true_params, atol=0.1))


if __name__ == "__main__":
    unittest.main()