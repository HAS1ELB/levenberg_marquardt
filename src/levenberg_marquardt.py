import numpy as np


def levenberg_marquardt(func, jacobian, x0, t, y_data, max_iter=100, tol=1e-6, lambda_init=0.01, model_type="exponential"):
    """
    Implémentation de la méthode de Levenberg-Marquardt pour l'ajustement non linéaire.

    Parameters:
        func: Fonction modèle. Prend x (paramètres) et t (données indépendantes) comme entrées.
        jacobian: Fonction qui calcule la matrice Jacobienne. Prend x et t comme entrées.
        x0: Liste ou tableau des paramètres initiaux.
        t: Données indépendantes (ex: temps).
        y_data: Données observées à ajuster.
        max_iter: Nombre maximal d'itérations.
        tol: Tolérance pour la convergence (sur la norme de la mise à jour des paramètres).
        lambda_init: Facteur d'amortissement initial.
        model_type: Type du modèle (ex : "exponential", "polynomial", "sinusoidal").

    Returns:
        np.ndarray: Paramètres optimisés.
    """
    # Initialisation des paramètres
    x = np.array(x0, dtype=float)
    lambda_param = lambda_init  # Facteur d'amortissement (lambda)

    for iteration in range(max_iter):
        # Calcul des résidus (différence entre les données et le modèle)
        residuals = y_data - func(x, t, model_type)

        # Calcul de la matrice Jacobienne
        J = jacobian(x, t, model_type)

        # Construction des matrices H et g
        H = J.T @ J + lambda_param * np.eye(J.shape[1])  # H = J^T * J + λ * I
        g = J.T @ residuals  # g = J^T * r

        # Résolution du système pour la mise à jour des paramètres
        try:
            H_inv = np.linalg.inv(H)  # Inversion de la matrice Hessienne
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)  # Si H est non inversible, utiliser la pseudo-inverse

        delta_x = H_inv @ g
        x_new = x + delta_x

        # Vérification de la convergence
        if np.linalg.norm(delta_x) < tol:
            print(f"Convergence atteinte à l'itération {iteration}.")
            return x_new

        # Évaluation des résidus avec les nouveaux paramètres
        new_residuals = y_data - func(x_new, t, model_type)

        # Mise à jour de lambda : Réduction si amélioration, sinon augmentation
        if np.linalg.norm(new_residuals) < np.linalg.norm(residuals):
            lambda_param /= 10  # Réduction du facteur d'amortissement
            x = x_new  # Mise à jour des paramètres
        else:
            lambda_param *= 10  # Augmentation du facteur d'amortissement

    print("Maximum d'itérations atteint. Retour des derniers paramètres.")
    return x
