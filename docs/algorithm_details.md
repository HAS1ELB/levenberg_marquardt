
# Détails Théoriques : Méthode de Levenberg-Marquardt

## Introduction
La méthode de Levenberg-Marquardt (LM) est un algorithme d'optimisation numérique utilisé principalement pour résoudre des problèmes d'ajustement de courbes non linéaires. Elle est utilisée lorsque la fonction de coût est non linéaire et nécessite l'optimisation de plusieurs paramètres.

## Principe de fonctionnement
La méthode de Levenberg-Marquardt est une combinaison de deux autres méthodes :
1. **La méthode des moindres carrés (Gauss-Newton)**, qui est efficace quand l'approximation linéaire de la fonction cible est bonne.
2. **La méthode de gradient**, qui est utilisée lorsque la fonction de coût est mal conditionnée ou que la Gauss-Newton échoue à converger.

Le principe est de minimiser l'erreur entre les valeurs observées et les valeurs prédites par un modèle. La méthode itère à travers des mises à jour des paramètres en utilisant des informations de la fonction de coût et de sa dérivée.

## Formulation mathématique
Soit un modèle de la forme :
$$ y = f(x, t) $$
Où :
- \( y \) est le vecteur des valeurs observées.
- \( f(x, t) \) est le modèle qui dépend des paramètres \( x \) et des données indépendantes \( t \).
- \( x \) est le vecteur des paramètres à optimiser.

Les résidus \( r_i \) entre les observations et les valeurs prédites par le modèle sont donnés par :
$$ r_i = y_i - f(x, t_i) $$

La méthode cherche à minimiser la somme des carrés des résidus :
$$ S(x) = \sum_{i=1}^n r_i^2 $$

La mise à jour des paramètres se fait selon :
$$ x_{k+1} = x_k + \Delta x_k $$

### Résolution du système de Levenberg-Marquardt
La mise à jour des paramètres \( \Delta x_k \) se résout via le système linéaire :
$$ (J^T J + \lambda I) \Delta x_k = J^T r_k $$

Où :
- \( J \) est la matrice Jacobienne des dérivées partielles du modèle par rapport aux paramètres.
- \( \lambda \) est un facteur d'amortissement qui ajuste l'importance de la méthode de Gauss-Newton ou de gradient.

## Algorithme
L'algorithme fonctionne comme suit :
1. **Initialisation** :
   - Choisir des valeurs initiales pour \( x \) et \( \lambda \).
2. **Calcul des résidus** :
   - Calculer les résidus entre les données observées et les valeurs prédites par le modèle.
3. **Calcul de la Jacobienne** :
   - Calculer la matrice Jacobienne \( J \), qui contient les dérivées partielles du modèle par rapport aux paramètres.
4. **Calcul de la mise à jour des paramètres** :
   - Résoudre le système pour obtenir \( \Delta x \), la mise à jour des paramètres.
5. **Critère d'arrêt** :
   - Si la norme de la mise à jour des paramètres est inférieure à une tolérance donnée, l'algorithme converge et s'arrête.
   - Si les résidus sont améliorés, réduire \( \lambda \); sinon, augmenter \( \lambda \).
6. **Répéter** les étapes jusqu'à la convergence ou atteindre le nombre maximal d'itérations.

## Avantages
- La méthode de Levenberg-Marquardt est robuste et converge généralement rapidement lorsqu'elle est bien paramétrée.
- Elle peut gérer des problèmes d'optimisation non linéaire complexes avec plusieurs paramètres.

## Applications
- Ajustement de courbes et modélisation de données expérimentales.
- Estimation des paramètres dans les systèmes dynamiques et les modèles de prédiction.
- Traitement d'images, vision par ordinateur et apprentissage automatique.

## Conclusion
La méthode de Levenberg-Marquardt est un algorithme puissant et largement utilisé pour résoudre des problèmes d'optimisation non linéaire. Son efficacité en fait un choix privilégié pour de nombreux domaines scientifiques et d'ingénierie.
