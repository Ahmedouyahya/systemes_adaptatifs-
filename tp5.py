import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1️ CHARGEMENT DES DONNÉES
# ---------------------------
print("\n[1] Chargement des données...")
ratings = pd.read_csv('ratings_tp5.csv', sep=',', na_values=[''])
movies = pd.read_csv('movies_tp5.csv', sep=',')

print(f"• {len(ratings)} évaluations chargées")
print(f"• {len(movies)} films chargés")

# Vérifier les colonnes
print("\nColonnes des fichiers :")
print("Ratings :", ratings.columns)
print("Movies :", movies.columns)

# ---------------------------
# 2️ FUSION DES DONNÉES
# ---------------------------
print("\n[2] Fusion des données...")
df = pd.merge(ratings, movies, on='movieId', how='inner')
print(f"• Dataset fusionné : {df.shape[0]} lignes")

# ---------------------------
# 3️ MATRICE UTILISATEUR-FILM
# ---------------------------
print("\n[3] Création de la matrice utilisateur-film...")
matrix = df.pivot_table(index='title', columns='userId', values='rating')

print(f"• Dimensions : {matrix.shape[0]} films x {matrix.shape[1]} utilisateurs")
print(f"• Taux de remplissage : {(1 - matrix.isna().mean().mean()):.1%}")

# ---------------------------
# 4️ NORMALISATION DES NOTES
# ---------------------------
print("\n[4] Normalisation des notes...")
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis=0)

# ---------------------------
# 5️ CALCUL DES SIMILARITÉS (Pearson & Cosinus)
# ---------------------------
print("\n[5] Calcul des similarités...")

# Pearson
print("• Calcul des corrélations de Pearson...")
similarity_pearson = matrix_norm.T.corr()
print("• Matrice de similarité Pearson (extrait) :")
print(similarity_pearson.head())

# Cosinus
print("• Calcul des similarités cosinus...")
matrix_norm_zero = matrix_norm.fillna(0)
similarity_cosine = pd.DataFrame(
    cosine_similarity(matrix_norm_zero),
    index=matrix_norm.index,
    columns=matrix_norm.index
)

print("• Matrice de similarité Cosinus (extrait) :")
print(similarity_cosine.head())

# ---------------------------
# 6️ PRÉDICTION DES NOTES
# ---------------------------
def predict_rating(user_id, movie, similarity_matrix, n_neighbors=2):
    """Prédit la note d'un utilisateur pour un film donné."""
    if movie not in similarity_matrix:
        return matrix.mean().mean()  # Retourne la moyenne globale si film inconnu

    # Trouver les films similaires
    similar_movies = similarity_matrix[movie].dropna().sort_values(ascending=False)

    # Garder les N voisins les plus proches
    top_neighbors = similar_movies.head(n_neighbors)

    # Films déjà notés par l'utilisateur
    rated_movies = matrix_norm[user_id].dropna()

    # Vérifier si des voisins existent
    common_movies = rated_movies.index.intersection(top_neighbors.index)
    if common_movies.empty:
        return matrix.mean().mean()  # Retourner la moyenne globale

    # Calculer la prédiction
    top_neighbors = top_neighbors.loc[common_movies]
    pred_norm = np.average(rated_movies[common_movies], weights=top_neighbors)

    # Dé-normalisation
    pred = pred_norm + matrix.mean(axis=1)[movie]
    return round(pred, 2)

# ---------------------------
# 7️ APPROCHE HYBRIDE (Combinaison Pearson & Cosinus)
# ---------------------------
def hybrid_prediction(user_id, movie, alpha=0.5, n_neighbors=2):
    """Prédit la note en combinant Pearson et Cosinus."""
    pred_p = predict_rating(user_id, movie, similarity_pearson, n_neighbors)
    pred_c = predict_rating(user_id, movie, similarity_cosine, n_neighbors)
    
    # Combinaison hybride
    return round(alpha * pred_p + (1 - alpha) * pred_c, 2)

# ---------------------------
# 8️ RECOMMANDATION POUR L'UTILISATEUR
# ---------------------------
def generate_recommendations(user_id, alpha=0.5, n_neighbors=2, n_reco=3):
    """Génère une liste de recommandations pour un utilisateur donné."""
    print(f"\n=== Résultats Hybrides pour l'utilisateur {user_id} ===")

    # Films non évalués
    unwatched_movies = matrix[user_id][matrix[user_id].isna()].index

    # Générer les prédictions
    predictions = [
        (movie, hybrid_prediction(user_id, movie, alpha, n_neighbors))
        for movie in unwatched_movies
    ]

    # Trier par note prédite et retourner les N meilleurs films
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_reco]

    # Afficher les recommandations
    for film, note in recommendations:
        print(f" {film} : {note}/5")

    return recommendations

# ---------------------------
# 9 AFFICHER LES RÉSULTATS
# ---------------------------
user_id_test = 4  # ID de l'utilisateur à recommander
recommendations = generate_recommendations(user_id=user_id_test, alpha=0.5, n_neighbors=2, n_reco=3)
