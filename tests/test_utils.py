import unittest
import numpy as np

from src.utils import (
    generate_synthetic_data,
    mean_squared_error,
    calculate_residuals
)
from src.main import func


class TestUtils(unittest.TestCase):
    def test_generate_synthetic_data(self):
        # Générer des données synthétiques avec un bruit faible
        t = np.linspace(0, 10, 100)
        true_params = [2.5, 1.3]
        y_data = generate_synthetic_data(func, true_params, t, noise_std=0.1)

        # Vérifier la forme des données générées
        self.assertEqual(len(y_data), len(t))

        # Vérifier que les données sont proches du modèle réel avec un bruit ajouté
        y_true = func(true_params, t)
        self.assertTrue(np.allclose(y_data[:10], y_true[:10], atol=0.2))

    def test_mean_squared_error(self):
        # Générer des données synthétiques
        t = np.linspace(0, 10, 100)
        true_params = [2.5, 1.3]
        y_data = generate_synthetic_data(func, true_params, t, noise_std=0.1)
        
        # Calculer l'erreur quadratique moyenne
        y_pred = func([2.5, 1.3], t)
        mse = mean_squared_error(y_data, y_pred)

        # Tester si MSE est une valeur positive et raisonnable
        self.assertGreaterEqual(mse, 0)
        self.assertLess(mse, 1)

    def test_calculate_residuals(self):
        # Générer des données synthétiques
        t = np.linspace(0, 10, 100)
        true_params = [2.5, 1.3]
        y_data = generate_synthetic_data(func, true_params, t, noise_std=0.1)

        # Calculer les résidus
        y_pred = func([2.5, 1.3], t)
        residuals = calculate_residuals(y_data, y_pred)

        # Tester si les résidus ont la même forme que y_data
        self.assertEqual(len(residuals), len(y_data))

        # Tester si la somme des résidus est proche de zéro (car y_pred est calculé avec les vrais paramètres)
        self.assertAlmostEqual(np.sum(residuals), 0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
