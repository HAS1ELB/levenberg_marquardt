import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.levenberg_marquardt import levenberg_marquardt
from src.utils import generate_synthetic_data
import matplotlib.pyplot as plt


# Fonction modèle et jacobienne pour les tests
def func(x, t):
    return x[0] * np.exp(-x[1] * t)


def jacobian(x, t):
    J = np.zeros((len(t), len(x)))
    J[:, 0] = np.exp(-x[1] * t)
    J[:, 1] = -x[0] * t * np.exp(-x[1] * t)
    return J


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
