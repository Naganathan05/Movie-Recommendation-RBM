import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
import requests
import urllib.parse  # Add this for URL encoding

def download_and_extract_dataset():
    """Download and extract the MovieLens 1M dataset if not already available"""
    if not os.path.exists('ml-1m'):
        print("Downloading MovieLens 1M dataset...")
        url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
        zip_path = 'ml-1m.zip'
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully.")
    else:
        print("MovieLens 1M dataset already exists.")

def load_and_preprocess_data():
    """Load and preprocess the MovieLens 1M dataset"""
    # Load MovieLens 1M dataset
    movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', 
                          engine='python', encoding='latin-1',
                          names=['movie_id', 'title', 'genres'])
    
    ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', 
                           engine='python', encoding='latin-1',
                           names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    users_df = pd.read_csv('ml-1m/users.dat', sep='::', 
                         engine='python', encoding='latin-1',
                         names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])

    print(f"Total users: {ratings_df['user_id'].nunique()}")
    print(f"Total movies: {ratings_df['movie_id'].nunique()}")
    print(f"Total ratings: {len(ratings_df)}")

    # Create user-item rating matrix
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')

    # Fill NA with -1 (to distinguish missing ratings from actual ratings)
    user_item_matrix = user_item_matrix.fillna(-1)

    # Convert to numpy array
    training_set = user_item_matrix.values

    # Normalize the ratings to [0,1] range while keeping -1 for missing ratings
    for i in range(training_set.shape[0]):
        rated_items = training_set[i, :] >= 0
        if np.sum(rated_items) > 0:
            training_set[i, rated_items] = training_set[i, rated_items] / 5.0

    # Split into training and testing sets
    train_data, test_data = train_test_split(training_set, test_size=0.2, random_state=42)

    return train_data, test_data, user_item_matrix, movies_df

def search_movies(movies_df, query):
    """Search for movies by title"""
    # Case-insensitive search in movie titles
    if query:
        matching_movies = movies_df[movies_df['title'].str.contains(query, case=False)]
        return matching_movies
    return pd.DataFrame()

def fetch_movie_poster(movie_title, api_key):
    """Fetch movie poster from OMDB API"""
    # Cache posters to avoid redundant API calls
    if 'poster_cache' not in st.session_state:
        st.session_state.poster_cache = {}
    
    # Check if poster is already in cache
    if movie_title in st.session_state.poster_cache:
        return st.session_state.poster_cache[movie_title]
    
    # Remove year from movie title (format: "Title (Year)")
    search_title = movie_title
    if "(" in movie_title and ")" in movie_title:
        search_title = movie_title.split("(")[0].strip()
    
    # URL encode the search title to handle special characters
    encoded_title = urllib.parse.quote(search_title)
    
    # Make API request
    try:
        # Use the encoded title in the API request
        url = f"http://www.omdbapi.com/?t={encoded_title}&apikey={api_key}"
        response = requests.get(url)
        
        # Check if the response is valid
        if response.status_code == 200:
            data = response.json()
            
            # # Debug: Print response data to console
            # print(f"API Response for '{search_title}': {data}")
            
            if data.get('Response') == 'True' and 'Poster' in data and data['Poster'] != 'N/A':
                poster_url = data['Poster']
            else:
                # If no poster found, provide a placeholder with the movie title
                encoded_placeholder = urllib.parse.quote(f"No Poster: {search_title}")
                poster_url = "assets/placeholder.jpg"
        else:
            print(f"API Error: Status code {response.status_code} for movie '{search_title}'")
            poster_url = "assets/placeholder.jpg"
        
        # Store in cache
        st.session_state.poster_cache[movie_title] = poster_url
        return poster_url
    except Exception as e:
        print(f"Error fetching poster for '{search_title}': {e}")
        return f"https://via.placeholder.com/150x225?text=Error:{type(e).__name__}"