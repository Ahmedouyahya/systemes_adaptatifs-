import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement des données
ratings = pd.read_csv('ratings_tp5.csv', sep=',', na_values=[''])
movies  = pd.read_csv('movies_tp5.csv', sep=',', na_values=[''])

# Fusion des données sur la colonne "movieId"
df = pd.merge(ratings, movies, on='movieId', how='inner')

# Création de la matrice utilisateur-film
# Ici, nous plaçons les utilisateurs en lignes et les films en colonnes
R = df.pivot_table(index='userId', columns='movieId', values='rating')
print("Dimensions de la matrice utilisateur-film :", R.shape)

# Remplacer les valeurs manquantes par 0 pour la SVD
R_filled = R.fillna(0)

# Convertir le DataFrame en tableau NumPy
R_filled_matrix = R_filled.values

# 2. Décomposition en valeurs singulières (SVD)
# Choix d'un rang k, par exemple k = 50
k = 50
U, sigma, Vt = svds(R_filled_matrix, k=k)
sigma = np.diag(sigma)

# Reconstruction de la matrice des prédictions
R_pred = np.dot(np.dot(U, sigma), Vt)
preds_df = pd.DataFrame(R_pred, index=R.index, columns=R.columns)

# 3. Fonction de recommandation basée sur SVD
def recommend_movies(user_id, preds_df, movies, original_ratings, num_recommendations=5):
    """
    Génère des recommandations pour l'utilisateur 'user_id' en utilisant la décomposition SVD.
    """
    # Extraire les prédictions pour l'utilisateur
    user_pred = preds_df.loc[user_id]
    
    # Récupérer les films déjà notés par l'utilisateur
    rated_movies = original_ratings[original_ratings.userId == user_id]['movieId']
    
    # Filtrer les films non notés
    recommendations = movies[~movies['movieId'].isin(rated_movies)]
    
    # Ajouter la note prédite
    recommendations = recommendations.copy()
    recommendations['predicted_rating'] = recommendations['movieId'].apply(lambda x: user_pred.get(x, 0))
    
    # Trier par note prédite décroissante
    recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
    return recommendations.head(num_recommendations)

# 4. Générer et afficher des recommandations pour un utilisateur (par exemple, l'utilisateur 4)
user_id = 4
reco = recommend_movies(user_id, preds_df, movies, ratings, num_recommendations=5)
print("\nRecommandations pour l'utilisateur", user_id, ":")
print(reco[['movieId', 'title', 'predicted_rating']])
