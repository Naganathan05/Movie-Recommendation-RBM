from models.rbm import RBM
import numpy as np
import pandas as pd
import torch

def recommend_movies(rbm, user_ratings, user_item_matrix, movies_df, num_recommendations=10):
    """Generate movie recommendations for a specific user based on their ratings."""
    # Create a user vector with -1 for all movies
    new_user_vector = np.full(user_item_matrix.shape[1], -1.0)
    
    # Fill in the ratings provided by the user
    for movie_id, rating in user_ratings.items():
        try:
            col_idx = user_item_matrix.columns.get_loc(movie_id)
            new_user_vector[col_idx] = rating / 5.0  # Normalize rating to 0-1 range
        except KeyError:
            continue
    
    # Reshape and convert to tensor
    new_user_tensor = torch.FloatTensor(new_user_vector.reshape(1, -1))
    
    # Get the reconstructed ratings
    reconstructed_ratings = rbm(new_user_tensor).detach().numpy().flatten()
    
    # Get indices of movies the user hasn't rated yet
    unrated_indices = np.where(new_user_vector < 0)[0]
    
    # Get top recommended movie indices
    recommended_indices = unrated_indices[np.argsort(-reconstructed_ratings[unrated_indices])[:num_recommendations]]
    
    # Map indices to movie IDs
    movie_ids = user_item_matrix.columns[recommended_indices].tolist()
    
    # Get movie titles
    recommended_movies = movies_df[movies_df['movie_id'].isin(movie_ids)][['movie_id', 'title', 'genres']]
    recommended_movies['predicted_rating'] = reconstructed_ratings[recommended_indices] * 5  # Scale back to 1-5 rating
    
    return recommended_movies.sort_values('predicted_rating', ascending=False)

# Add the missing function for compatibility with the app
get_recommendations_for_new_user = recommend_movies