
# Levenberg-Marquardt Project

Ce projet implémente l'algorithme de **Levenberg-Marquardt** pour l'ajustement non linéaire des courbes. Il inclut un script principal pour exécuter l'algorithme, des fonctions auxiliaires pour générer des données et afficher les résultats, ainsi que des tests unitaires pour assurer la robustesse du code.

## Structure du Projet

Le projet est organisé de la manière suivante :

```
levenberg-marquardt-project/
│
├── src/
│   ├── main.py               # Script principal pour exécuter l'algorithme.
│   ├── levenberg_marquardt.py  # Algorithme principal.
│   ├── utils.py              # Fonctions auxiliaires (optionnelles).
│   └── __init__.py           # Indique que c'est un package.
│
├── tests/
│   ├── test_algorithm.py     # Tests pour l'algorithme.
│   └── test_utils.py         # Tests pour les fonctions utilitaires.
│
├── docs/
│   ├── README.md             # Documentation du projet.
│   └── algorithm_details.md  # Détails théoriques.
│
├── requirements.txt          # Liste des dépendances.
└── .vscode/
    └── launch.json           # Configuration pour le débogage dans VSCode.
```

## Installation

### Prérequis

Avant de commencer, assurez-vous d'avoir installé **Python 3.x** et **pip** sur votre machine.

### Étapes d'installation

1. Clonez ce repository sur votre machine locale :
    ```bash
    git clone https://github.com/votre-utilisateur/levenberg-marquardt-project.git
    ```
2. Accédez au répertoire du projet :
    ```bash
    cd levenberg-marquardt-project
    ```
3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### Exécution du script principal

Le script `main.py` contient l'exécution de l'algorithme. Vous pouvez l'exécuter directement en ligne de commande pour ajuster un modèle avec des données synthétiques :

```bash
python src/main.py
```

Le programme générera des données avec bruit, ajustera un modèle en utilisant l'algorithme de Levenberg-Marquardt, et affichera les résultats.

### Tests

Le projet comprend des tests unitaires pour vérifier l'exactitude de l'algorithme et des fonctions utilitaires. Vous pouvez exécuter les tests avec **pytest** :

```bash
pytest tests/
```

### Documentation

La documentation théorique sur la méthode de Levenberg-Marquardt se trouve dans le fichier `docs/algorithm_details.md`. Ce fichier détaille les principes mathématiques et les étapes de l'algorithme.

## Débogage dans VSCode

Le projet inclut une configuration de débogage pour VSCode. Pour l'utiliser, ouvrez le projet dans VSCode et lancez le débogueur avec la configuration présente dans le fichier `.vscode/launch.json`.

## Contributeurs

- **Votre nom** : Développeur principal.

## Licence

Ce projet est sous licence **MIT**. Vous pouvez librement utiliser et modifier ce code, à condition de conserver cette mention.
