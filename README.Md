# Projet KNNClassifier

# Auteur : Mohamadou Lamine NIANG

Ce projet contient une implémentation simple de l'algorithme des k-plus proches voisins (KNN) en utilisant Python. L'algorithme est utilisé pour la classification des données.

## Fonctionnalités

- Implémentation de base de l'algorithme KNN en utilisant Python.
- Prend en charge la spécification du nombre de voisins (k).
- Prise en charge de la distance euclidienne pour calculer la similarité entre les points.
- Capable de faire des prédictions sur de nouveaux exemples.

## Utilisation

1. Clonez ce dépôt sur votre machine locale.
2. Utilisez le fichier `KNNClassifier.py` pour instancier l'algorithme KNN et effectuer des prédictions sur vos données.
3. Assurez-vous d'avoir les dépendances requises installées, telles que NumPy.


# Exemple de création d'une instance de KNNClassifier avec k=3
knn = KNNClassifier(k=3)

# Exemple de l'ensemble des  données

data, target = make_classification(n_samples=10000, n_features=4, n_classes=2)


# Division  des données d'entraitement et de test 
X_train = data[:8000]
y_train = target[:8000]
X_test = data[8000:]
y_test = target[8000:]


# Entraînement de l'algorithme
knn.fit(X_train, y_train)

# Prédictions
predictions = knn.predict(X_test)
print(predictions)  # Output: [1]
