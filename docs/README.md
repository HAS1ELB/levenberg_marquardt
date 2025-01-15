# Levenberg-Marquardt Project 🚀

Ce projet implémente l'algorithme **Levenberg-Marquardt** pour l'ajustement non linéaire des courbes, une méthode robuste et rapide utilisée dans de nombreux domaines scientifiques et d'ingénierie.

---

## 🌟 Fonctionnalités principales

- **Optimisation non linéaire** : Implémentation complète de l'algorithme pour ajuster des modèles complexes.
- **Visualisation des résultats** : Graphiques interactifs pour mieux comprendre les ajustements.
- **Tests unitaires** : Garantir la robustesse et la fiabilité du code.

---

## 📂 Structure du projet

```
has1elb-levenberg_marquardt/
├── docs/                 # Documentation et théorie
│   ├── README.md         # Documentation principale
│   └── algorithm_details.md  # Explications théoriques de l'algorithme
├── src/                  # Code source
│   ├── main.py           # Script principal
│   ├── gui.py            # Interface utilisateur
│   ├── utils.py          # Fonctions utilitaires
│   ├── levenberg_marquardt.py # Algorithme principal
│   └── __init__.py
├── tests/                # Tests unitaires
│   ├── test_algorithm.py # Tests pour l'algorithme
│   ├── test_utils.py     # Tests pour les utilitaires
│   └── __init__.py
├── requirements.txt      # Dépendances nécessaires
└── .vscode/              # Configuration VSCode
```

---

## ⚙️ Installation

### Prérequis

- Python 3.x
- pip

### Étapes

1. Clonez ce repository :
   ```bash
   git clone https://github.com/has1elb/levenberg-marquardt-project.git
   ```
2. Accédez au dossier :
   ```bash
   cd levenberg-marquardt-project
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Utilisation

### Exécution principale

Lancez le script principal pour voir l'algorithme en action :

```bash
python src/main.py
```

### Interface utilisateur

Utilisez une interface interactive avec **Streamlit** :

```bash
streamlit run src/gui.py
```

### Tests

Vérifiez l'intégrité du code avec **pytest** :

```bash
pytest tests/
```

---

## 📖 Documentation

- **Détails théoriques** : [algorithm_details.md](docs/algorithm_details.md)
- **Structure et guide** : Ce fichier README.

---

## 👥 Contributeurs

- **EL BAHRAOUI HASSAN**
- **EL BACHAR WALID**
- **DEHBI KAMAL**
- **MALEK SAMI**

---

## 🌍 Connectez avec nous

Contributions, questions et suggestions sont les bienvenues ! 🎉
