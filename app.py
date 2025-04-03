import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import sys
import warnings

# Add directory to path so we can import from our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from models.rbm import RBM
from utils.data_processor import load_and_preprocess_data, download_and_extract_dataset, search_movies, fetch_movie_poster
from utils.recommender import get_recommendations_for_new_user

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
)

# Try to load CSS, handle error if not found
try:
    with open('assets/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found. Using default styling.")
    
def load_data():
    """Load dataset and preprocess it"""
    # Download dataset if needed
    download_and_extract_dataset()
    # Load and preprocess data
    return load_and_preprocess_data()

def get_model(nb_movies, nb_hidden, learning_rate=0.005):
    """Load existing model or return None if not available"""
    model_path = "rbm_model.pt"
    
    if os.path.exists(model_path):
        try:
            model = RBM(nb_movies, nb_hidden, learning_rate)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    # If model doesn't exist or couldn't be loaded
    return None

def train_rbm(rbm, train_data, nb_epoch=10, batch_size=64, k_steps=5):
    """Train the RBM model with progress indicators"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    nb_users = train_data.shape[0]
    training_loss_history = []

    for epoch in range(1, nb_epoch + 1):
        train_loss = 0.0
        s = 0.0
        
        # Process in batches
        for id_user in range(0, nb_users - batch_size, batch_size):
            v0 = torch.FloatTensor(train_data[id_user:id_user+batch_size])
            
            # Perform contrastive divergence and update weights
            loss = rbm.contrastive_divergence(v0, k=k_steps)
            
            train_loss += loss.item()
            s += 1.0
        
        # Average loss for this epoch
        epoch_loss = train_loss/s if s > 0 else 0
        training_loss_history.append(epoch_loss)
        
        # Update progress
        progress = epoch / nb_epoch
        progress_bar.progress(progress)
        status_text.text(f'Epoch: {epoch}/{nb_epoch} - Loss: {epoch_loss:.4f}')

    status_text.text('Training complete!')
    return training_loss_history

def save_model(model, filepath='rbm_model.pt'):
    """Save the trained RBM model to a file"""
    try:
        torch.save(model.state_dict(), filepath)
        st.success(f"Model saved successfully to {filepath}")
    except Exception as e:
        st.error(f"Error saving model: {e}")

def plot_training_loss(loss_history, nb_epoch):
    """Plot the training loss over epochs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, nb_epoch + 1), loss_history)
    ax.set_title('Training Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    st.pyplot(fig)

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("A recommendation system using Restricted Boltzmann Machines (RBM)")
    
    # Hardcoded API key - no user input needed
    api_key = "4e634d4f"  # Hardcoded OMDB API key
    
    # Load data - use st.session_state to avoid reloading
    if 'data_loaded' not in st.session_state:
        with st.spinner('Loading data...'):
            try:
                train_data, test_data, user_item_matrix, movies = load_data()
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                st.session_state.user_item_matrix = user_item_matrix
                st.session_state.movies = movies
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
    else:
        train_data = st.session_state.train_data
        test_data = st.session_state.test_data
        user_item_matrix = st.session_state.user_item_matrix
        movies = st.session_state.movies
    
    nb_movies = train_data.shape[1]
    nb_hidden = 150
    learning_rate = 0.005
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Get Recommendations", "Train Model", "About"])
    
    with tab1:
        st.header("Get Movie Recommendations")
        
        # Load the model - avoid using cache for pytorch models
        rbm = get_model(nb_movies, nb_hidden, learning_rate)
        
        if rbm is None:
            st.info("Please go to the 'Train Model' tab to train a model first.")
        else:
            # Session state for storing user ratings
            if 'user_ratings' not in st.session_state:
                st.session_state.user_ratings = {}
            
            # Use 3 columns with appropriate widths
            col1, col2, col3 = st.columns([2, 1, 3])
            
            with col1:
                st.subheader("Find Movies")
                query = st.text_input("Search for movies:")
                matching_movies = search_movies(movies, query)
                
                if not matching_movies.empty:
                    # Display movies with posters in a grid layout with 3 movies per row
                    num_cols = 3  # Number of columns in the grid
                    # Limit number of displayed search results
                    max_results = 12
                    limited_movies = matching_movies.head(max_results)
                    
                    movie_rows = [limited_movies['title'].tolist()[i:i+num_cols] 
                                 for i in range(0, len(limited_movies), num_cols)]
                    
                    for row in movie_rows:
                        movie_cols = st.columns(num_cols)
                        for i, movie_title in enumerate(row):
                            if i < len(movie_cols):
                                movie_id = limited_movies[limited_movies['title'] == movie_title]['movie_id'].values[0]
                                
                                with movie_cols[i]:
                                    poster_url = fetch_movie_poster(movie_title, api_key)
                                    st.image(poster_url, width=100)  # Smaller poster
                                    st.write(f"**{movie_title[:25]}**" + ("..." if len(movie_title) > 25 else ""))
                                    rating = st.slider("Rate:", 1, 5, 3, key=f"rate_{movie_id}", 
                                                      label_visibility="collapsed")
                                    if st.button("Rate", key=f"btn_{movie_id}", use_container_width=True):
                                        st.session_state.user_ratings[movie_id] = rating
                                        st.success(f"Rated: {rating}/5", icon="✅")
                                        st.rerun()
                    
                    if len(matching_movies) > max_results:
                        st.info(f"Showing top {max_results} results. Refine your search for more specific movies.")
            
            with col2:
                st.subheader("Your Ratings")
                if st.session_state.user_ratings:
                    # Create a scrollable area for ratings
                    with st.container():
                        for movie_id, rating in st.session_state.user_ratings.items():
                            try:
                                movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
                                st.write(f"**{movie_title[:20]}{'...' if len(movie_title) > 20 else ''}**: {'⭐' * rating}")
                            except IndexError:
                                st.write(f"Movie ID {movie_id}: {rating}/5")
                    
                    if st.button("Clear Ratings", use_container_width=True):
                        st.session_state.user_ratings = {}
                        st.rerun()
                else:
                    st.info("No movies rated yet")
            
            with col3:
                st.subheader("Recommendations")
                if st.session_state.user_ratings:
                    num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
                    
                    if st.button("Get Recommendations", use_container_width=True):
                        with st.spinner('Generating recommendations...'):
                            try:
                                recommendations = get_recommendations_for_new_user(
                                    rbm, st.session_state.user_ratings, user_item_matrix, 
                                    movies, num_recommendations
                                )
                                
                                st.subheader("Top Picks For You:")
                                
                                # Display recommendations in a grid with posters - 3 columns
                                num_rec_cols = 3
                                rec_rows = [(i, row) for i, (_, row) in enumerate(recommendations.iterrows(), 1)]
                                rec_rows = [rec_rows[i:i+num_rec_cols] for i in range(0, len(rec_rows), num_rec_cols)]
                                
                                for row in rec_rows:
                                    cols = st.columns(num_rec_cols)
                                    for j, (i, rec) in enumerate(row):
                                        if j < len(cols):
                                            with cols[j]:
                                                poster_url = fetch_movie_poster(rec['title'], api_key)
                                                st.image(poster_url, width=120)
                                                st.write(f"**{i}. {rec['title'][:25]}**" + 
                                                         ("..." if len(rec['title']) > 25 else ""))
                                                st.write(f"⭐ {rec['predicted_rating']:.1f}/5")
                            except Exception as e:
                                st.error(f"Error generating recommendations: {e}")
                else:
                    st.info("Please rate some movies to get recommendations.")
                    
                    # Show example empty recommendation cards for visual layout
                    st.write("Sample recommendations will appear here:")
                    sample_cols = st.columns(3)
                    for i, col in enumerate(sample_cols):
                        with col:
                            st.image("assets/placeholder.jpg", width=120)
                            st.write("**Example Movie**")
                            st.write("⭐ ?/5")
    
    with tab2:
        st.header("Train Model")
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        with col1:
            nb_epoch = st.slider("Number of epochs:", 1, 20, 10)
            batch_size = st.slider("Batch size:", 16, 128, 64)
        with col2:
            k_steps = st.slider("CD steps:", 1, 10, 5)
            learning_rate = st.slider("Learning rate:", 0.001, 0.01, 0.005, format="%.4f")
        
        if st.button("Train New Model"):
            with st.spinner("Training model... This may take a while."):
                try:
                    rbm = RBM(nb_movies, nb_hidden, learning_rate)
                    training_loss_history = train_rbm(rbm, train_data, nb_epoch, batch_size, k_steps)
                    save_model(rbm, "rbm_model.pt")
                    
                    st.subheader("Training Results")
                    plot_training_loss(training_loss_history, nb_epoch)
                except Exception as e:
                    st.error(f"Error during model training: {e}")
    
    with tab3:
        st.header("About This App")
        st.write("""
        This movie recommendation system uses a Restricted Boltzmann Machine (RBM) to learn user
        preferences and recommend movies based on your ratings.
        
        ## How it works:
        1. The system is trained on the MovieLens 1M dataset containing 1 million ratings from 6,000 users on 4,000 movies.
        2. The RBM learns patterns in user ratings to understand preferences.
        3. When you rate movies, the system predicts how you would rate other movies you haven't seen.
        4. Movies with the highest predicted ratings are recommended to you.
        
        ## Technologies used:
        - PyTorch for the RBM implementation
        - Streamlit for the user interface
        - Python for data processing
        
        This project demonstrates the application of unsupervised learning in recommendation systems.
        """)

if __name__ == "__main__":
    main()