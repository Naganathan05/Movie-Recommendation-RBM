import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import sys
import warnings
import torch.nn as nn
import torch.optim as optim
import time  # Add time import

# Add directory to path so we can import from our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from models.rbm import RBM
from models.autoencoder import Autoencoder  # Add this import
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

def get_model(model_type, nb_movies, nb_hidden, learning_rate=0.005):
    """Load existing model or return None if not available"""
    if model_type == "rbm":
        model_path = "rbm_model.pt"
        
        if os.path.exists(model_path):
            try:
                model = RBM(nb_movies, nb_hidden, learning_rate)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                return model
            except Exception as e:
                st.error(f"Error loading RBM model: {e}")
    
    elif model_type == "autoencoder":
        model_path = "autoencoder_model.pt"
        
        if os.path.exists(model_path):
            try:
                # We need to use the exact same hidden dimensions as when training
                # For Autoencoder, we need to store the hidden_dim in the model file
                # or use a fixed hidden_dim
                
                # Try to load saved model parameters to determine dimensions
                saved_state = torch.load(model_path)
                # Extract dimensions from the first layer weight
                if "encoder.0.weight" in saved_state:
                    # The second dimension is always nb_movies
                    actual_hidden_dim = saved_state["encoder.0.weight"].shape[0] // 2
                    
                    # Create model with the same dimensions as the saved model
                    model = Autoencoder(nb_movies, actual_hidden_dim)
                    model.load_state_dict(saved_state)
                    model.eval()
                    return model
                else:
                    raise ValueError("Saved model has unexpected format")
                    
            except Exception as e:
                st.error(f"Error loading Autoencoder model: {e}")
                # You might want to delete the corrupted model file
                if st.button("Delete corrupted model and retrain?"):
                    try:
                        os.remove(model_path)
                        st.success("Old model removed. Please retrain.")
                    except:
                        st.error("Failed to remove old model file.")
    
    # If model doesn't exist or couldn't be loaded
    return None

def save_model(model, filepath, model_type="rbm"):
    """Save the trained model to a file"""
    try:
        torch.save(model.state_dict(), filepath)
        st.success(f"{model_type.upper()} model saved successfully to {filepath}")
    except Exception as e:
        st.error(f"Error saving {model_type} model: {e}")

def display_recommendations(recommendations, create_columns=True):
    """Display recommendations in a grid with posters
    
    Args:
        recommendations: DataFrame with movie recommendations
        create_columns: If False, doesn't create columns (for nested contexts)
    """
    num_rec_cols = 3
    rec_rows = [(i, row) for i, (_, row) in enumerate(recommendations.iterrows(), 1)]
    rec_rows = [rec_rows[i:i+num_rec_cols] for i in range(0, len(rec_rows), num_rec_cols)]
    
    for row in rec_rows:
        if create_columns:
            cols = st.columns(num_rec_cols)
            for j, (i, rec) in enumerate(row):
                if j < len(cols):
                    with cols[j]:
                        poster_url = fetch_movie_poster(rec['title'], st.session_state.api_key)
                        st.image(poster_url, width=120)
                        st.write(f"**{i}. {rec['title'][:25]}**" + 
                                 ("..." if len(rec['title']) > 25 else ""))
                        st.write(f"⭐ {rec['predicted_rating']:.1f}/5")
        else:
            # Display without creating more columns (for nested contexts)
            for j, (i, rec) in enumerate(row):
                poster_url = fetch_movie_poster(rec['title'], st.session_state.api_key)
                st.image(poster_url, width=80)  # Smaller image
                st.write(f"**{i}. {rec['title'][:25]}**" + 
                         ("..." if len(rec['title']) > 25 else ""))
                st.write(f"⭐ {rec['predicted_rating']:.1f}/5")
                # Add separator between recommendations except last one
                if j < len(row) - 1:
                    st.markdown("---")

def train_autoencoder(train_data, nb_epoch, batch_size, hidden_dim, learning_rate):
    """Train the Autoencoder model with progress indicators"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    # Convert data to torch tensor
    train_tensor = torch.FloatTensor(train_data)
    
    # Create model, loss function, and optimizer
    model = Autoencoder(train_data.shape[1], hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Training loop
    training_loss_history = []
    
    for epoch in range(1, nb_epoch + 1):
        model.train()  # Set to training mode
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass - ignore missing ratings (those with -1)
            mask = data >= 0
            output = model(data)
            
            # Set output for missing ratings to -1 (to match input)
            output = output * mask.float()
            
            # Calculate loss only on non-missing ratings
            loss = criterion(output[mask], target[mask])
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        training_loss_history.append(epoch_loss)
        
        # Update progress
        progress = epoch / nb_epoch
        progress_bar.progress(progress)
        status_text.text(f'Epoch: {epoch}/{nb_epoch} - Loss: {epoch_loss:.4f}')
    
    # Store metrics in session_state
    st.session_state.autoencoder_metrics = {
        'loss_history': training_loss_history,
        'epochs': nb_epoch
    }
    st.session_state.autoencoder_train_time = time.time() - start_time
    
    status_text.text('Training complete!')
    
    # Save the trained model
    save_model(model, "autoencoder_model.pt", model_type="autoencoder")
    return training_loss_history

def train_rbm(rbm, train_data, nb_epoch=10, batch_size=64, k_steps=5):
    """Train the RBM model with progress indicators"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()  # Add time import at the top
    
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

    # Store metrics in session_state
    st.session_state.rbm_metrics = {
        'loss_history': training_loss_history,
        'epochs': nb_epoch
    }
    st.session_state.rbm_train_time = time.time() - start_time
    
    status_text.text('Training complete!')
    return training_loss_history

def plot_training_loss(loss_history, nb_epoch):
    """Plot the training loss over epochs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, nb_epoch + 1), loss_history)
    ax.set_title('Training Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    st.pyplot(fig)

def calculate_model_error(model, test_data, model_type="rbm"):
    """Calculate Mean Absolute Error for a model on test data"""
    # Convert to tensor
    test_tensor = torch.FloatTensor(test_data)
    
    # Get predictions
    if model_type.lower() == "rbm":
        predictions = model(test_tensor).detach().numpy()
    else:
        model.eval()
        with torch.no_grad():
            predictions = model(test_tensor).numpy()
    
    # Only consider actual ratings (>=0)
    error_sum = 0
    count = 0
    
    for i in range(test_data.shape[0]):
        # Get indices of actual ratings
        actual_indices = np.where(test_data[i] >= 0)[0]
        
        # Skip if no ratings
        if len(actual_indices) == 0:
            continue
        
        # Calculate absolute error for this user
        user_error = np.mean(np.abs(
            predictions[i, actual_indices] - test_data[i, actual_indices]
        ))
        
        error_sum += user_error
        count += 1
    
    # Average error across all users
    mae = error_sum / count if count > 0 else 0
    return mae * 5  # Scale back to original rating scale (1-5)

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("A recommendation system using Restricted Boltzmann Machines (RBM)")
    
    # Hardcoded API key - no user input needed
    api_key = "4e634d4f"  # Hardcoded OMDB API key
    st.session_state.api_key = api_key
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["Get Recommendations", "Train Model", "Model Comparison", "About"])
    
    with tab1:
        st.header("Get Movie Recommendations")
        
        # Load the models - avoid using cache for pytorch models
        rbm = get_model("rbm", nb_movies, nb_hidden, learning_rate)
        autoencoder = get_model("autoencoder", nb_movies, nb_hidden, learning_rate)
        
        if rbm is None and autoencoder is None:
            st.info("Please go to the 'Train Model' tab to train a model first.")
        else:
            # Session state for storing user ratings
            if 'user_ratings' not in st.session_state:
                st.session_state.user_ratings = {}
            
            # Use 3 columns with appropriate widths
            col1, col2, col3 = st.columns([2, 1, 3])
            
            # First column - Movie search and rating
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
            
            # Second column - User ratings
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
            
            # Third column - Recommendations with model selection
            with col3:
                st.subheader("Recommendations")
                if st.session_state.user_ratings:
                    num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
                    
                    # Add model selection options
                    model_type = st.radio(
                        "Select model for recommendations:",
                        ["RBM", "Autoencoder", "Compare Both"],
                        horizontal=True
                    )
                    
                    # Get recommendations button
                    if st.button("Get Recommendations", use_container_width=True):
                        with st.spinner('Generating recommendations...'):
                            try:
                                if model_type == "RBM" and rbm is not None:
                                    recommendations = get_recommendations_for_new_user(
                                        rbm, st.session_state.user_ratings, user_item_matrix, 
                                        movies, num_recommendations, model_type="rbm"
                                    )
                                    st.subheader("Top Picks Using RBM:")
                                    display_recommendations(recommendations)
                                    
                                elif model_type == "Autoencoder" and autoencoder is not None:
                                    recommendations = get_recommendations_for_new_user(
                                        autoencoder, st.session_state.user_ratings, user_item_matrix, 
                                        movies, num_recommendations, model_type="autoencoder"
                                    )
                                    st.subheader("Top Picks Using Autoencoder:")
                                    display_recommendations(recommendations)
                                    
                                elif model_type == "Compare Both":
                                    col_rbm, col_ae = st.columns(2)
                                    
                                    with col_rbm:
                                        if rbm is not None:
                                            rbm_recs = get_recommendations_for_new_user(
                                                rbm, st.session_state.user_ratings, user_item_matrix, 
                                                movies, num_recommendations, model_type="rbm"
                                            )
                                            st.subheader("RBM Recommendations:")
                                            # Pass create_columns=False to avoid nested columns
                                            display_recommendations(rbm_recs, create_columns=False)
                                        else:
                                            st.warning("RBM model not trained yet")
                                    
                                    with col_ae:
                                        if autoencoder is not None:
                                            ae_recs = get_recommendations_for_new_user(
                                                autoencoder, st.session_state.user_ratings, user_item_matrix, 
                                                movies, num_recommendations, model_type="autoencoder"
                                            )
                                            st.subheader("Autoencoder Recommendations:")
                                            # Pass create_columns=False to avoid nested columns
                                            display_recommendations(ae_recs, create_columns=False)
                                        else:
                                            st.warning("Autoencoder model not trained yet")
                                else:
                                    if model_type == "RBM":
                                        st.warning("Please train the RBM model first")
                                    else:
                                        st.warning("Please train the Autoencoder model first")
                                
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
        
        # Model selection
        model_to_train = st.radio(
            "Select model to train:",
            ["Restricted Boltzmann Machine (RBM)", "Autoencoder"],
            horizontal=True
        )
        
        if model_to_train == "Restricted Boltzmann Machine (RBM)":
            # Model parameters for RBM
            st.subheader("RBM Parameters")
            col1, col2 = st.columns(2)
            with col1:
                nb_epoch = st.slider("Number of epochs:", 1, 20, 10)
                batch_size = st.slider("Batch size:", 16, 128, 64)
            with col2:
                k_steps = st.slider("CD steps:", 1, 10, 5)
                learning_rate = st.slider("Learning rate:", 0.001, 0.01, 0.005, format="%.4f")
            
            if st.button("Train RBM Model"):
                with st.spinner("Training RBM model... This may take a while."):
                    try:
                        rbm = RBM(nb_movies, nb_hidden, learning_rate)
                        training_loss_history = train_rbm(rbm, train_data, nb_epoch, batch_size, k_steps)
                        save_model(rbm, "rbm_model.pt", model_type="rbm")
                        
                        st.subheader("Training Results")
                        plot_training_loss(training_loss_history, nb_epoch)
                    except Exception as e:
                        st.error(f"Error during RBM model training: {e}")
        
        else:
            # Model parameters for Autoencoder
            st.subheader("Autoencoder Parameters")
            col1, col2 = st.columns(2)
            with col1:
                ae_epochs = st.slider("Number of epochs:", 1, 50, 20)
                ae_batch_size = st.slider("Batch size:", 16, 256, 128)
            with col2:
                ae_hidden_dim = st.slider("Hidden dimension:", 8, 512, 128)
                ae_lr = st.slider("Learning rate:", 0.0001, 0.01, 0.001, format="%.5f")
                
            if st.button("Train Autoencoder Model"):
                with st.spinner("Training Autoencoder model... This may take a while."):
                    try:
                        training_loss_history = train_autoencoder(
                            train_data, ae_epochs, ae_batch_size, ae_hidden_dim, ae_lr
                        )
                        st.subheader("Training Results")
                        plot_training_loss(training_loss_history, ae_epochs)
                    except Exception as e:
                        st.error(f"Error during Autoencoder model training: {e}")
    
    with tab3:
        st.header("Model Comparison")
        
        # Check if models exist
        rbm_exists = os.path.exists("rbm_model.pt")
        autoencoder_exists = os.path.exists("autoencoder_model.pt")
        
        if not rbm_exists and not autoencoder_exists:
            st.info("Please train at least one model in the 'Train Model' tab to see comparisons.")
        else:
            # Store training metrics in session state
            if 'rbm_metrics' not in st.session_state:
                st.session_state.rbm_metrics = {'loss_history': [], 'epochs': 0}
            if 'autoencoder_metrics' not in st.session_state:
                st.session_state.autoencoder_metrics = {'loss_history': [], 'epochs': 0}
            
            # Loss curves section
            st.subheader("Training Loss Comparison")
            
            if len(st.session_state.rbm_metrics['loss_history']) == 0 and len(st.session_state.autoencoder_metrics['loss_history']) == 0:
                st.write("No training metrics available yet. Train models to see comparison.")
            else:
                # Create loss curve plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot RBM loss if available
                if len(st.session_state.rbm_metrics['loss_history']) > 0:
                    rbm_epochs = st.session_state.rbm_metrics['epochs']
                    ax.plot(range(1, rbm_epochs + 1), st.session_state.rbm_metrics['loss_history'], 
                            label='RBM', color='blue')
                
                # Plot Autoencoder loss if available
                if len(st.session_state.autoencoder_metrics['loss_history']) > 0:
                    ae_epochs = st.session_state.autoencoder_metrics['epochs']
                    ax.plot(range(1, ae_epochs + 1), st.session_state.autoencoder_metrics['loss_history'], 
                            label='Autoencoder', color='green')
                
                ax.set_title('Training Loss Comparison')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            # Performance metrics section
            st.subheader("Performance Metrics")
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**RBM Model**")
                if rbm_exists:
                    # Load RBM and evaluate
                    rbm = get_model("rbm", nb_movies, nb_hidden, learning_rate)
                    if 'rbm_mae' not in st.session_state:
                        with st.spinner("Calculating RBM metrics..."):
                            try:
                                # Calculate MAE on test data
                                mae = calculate_model_error(rbm, test_data, "rbm")
                                st.session_state.rbm_mae = mae
                            except Exception as e:
                                st.error(f"Error calculating RBM metrics: {e}")
                                st.session_state.rbm_mae = None
                    
                    if st.session_state.rbm_mae is not None:
                        st.metric("Mean Absolute Error", f"{st.session_state.rbm_mae:.4f}")
                    
                    # Display parameters
                    st.write(f"Hidden Units: {nb_hidden}")
                    st.write(f"Learning Rate: {learning_rate}")
                    if 'rbm_train_time' in st.session_state:
                        st.write(f"Training Time: {st.session_state.rbm_train_time:.2f} seconds")
                else:
                    st.info("RBM model not trained yet")
            
            with col2:
                st.write("**Autoencoder Model**")
                if autoencoder_exists:
                    # Load Autoencoder and evaluate
                    autoencoder = get_model("autoencoder", nb_movies, nb_hidden, learning_rate)
                    if 'autoencoder_mae' not in st.session_state:
                        with st.spinner("Calculating Autoencoder metrics..."):
                            try:
                                # Calculate MAE on test data
                                mae = calculate_model_error(autoencoder, test_data, "autoencoder")
                                st.session_state.autoencoder_mae = mae
                            except Exception as e:
                                st.error(f"Error calculating Autoencoder metrics: {e}")
                                st.session_state.autoencoder_mae = None
                    
                    if st.session_state.autoencoder_mae is not None:
                        st.metric("Mean Absolute Error", f"{st.session_state.autoencoder_mae:.4f}")
                    
                    # Display parameters from the model if available
                    if autoencoder is not None:
                        hidden_dim = autoencoder.encoder[2].in_features
                        st.write(f"Hidden Units: {hidden_dim}")
                    if 'autoencoder_train_time' in st.session_state:
                        st.write(f"Training Time: {st.session_state.autoencoder_train_time:.2f} seconds")
                else:
                    st.info("Autoencoder model not trained yet")
            
            # Sample user comparison
            if rbm_exists and autoencoder_exists:
                st.subheader("Recommendation Comparison")
                
                # Select a random user from the test set
                if st.button("Generate comparison on random user"):
                    with st.spinner("Generating comparison..."):
                        try:
                            # Select a random user
                            test_user_idx = np.random.randint(0, test_data.shape[0])
                            test_user = torch.FloatTensor(test_data[test_user_idx:test_user_idx+1])
                            
                            # Get actual non-negative ratings for evaluation
                            actual_indices = np.where(test_data[test_user_idx] >= 0)[0]
                            actual_ratings = test_data[test_user_idx, actual_indices]
                            
                            # Get RBM predictions
                            rbm_preds = rbm(test_user).detach().numpy().flatten()
                            
                            # Get Autoencoder predictions
                            autoencoder.eval()
                            with torch.no_grad():
                                ae_preds = autoencoder(test_user).numpy().flatten()
                            
                            # Calculate errors on visible ratings only
                            rbm_error = np.mean(np.abs(rbm_preds[actual_indices] - actual_ratings))
                            ae_error = np.mean(np.abs(ae_preds[actual_indices] - actual_ratings))
                            
                            # Create bar chart comparing the two errors
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.bar(['RBM', 'Autoencoder'], [rbm_error, ae_error])
                            ax.set_title('Model Error Comparison')
                            ax.set_ylabel('Mean Absolute Error')
                            st.pyplot(fig)
                            
                            # Show actual vs. predicted for a few movies
                            st.write("**Sample Predictions for User**")
                            
                            # Get movie ids for sampled actual ratings
                            movie_indices = actual_indices[:5]  # Just show 5 movies
                            movie_ids = user_item_matrix.columns[movie_indices].tolist()
                            movie_titles = [movies[movies['movie_id'] == mid]['title'].values[0] for mid in movie_ids]
                            
                            # Create dataframe for comparison
                            comparison_df = pd.DataFrame({
                                'Movie': movie_titles,
                                'Actual': actual_ratings[:5] * 5,  # Scale back to 1-5
                                'RBM Prediction': rbm_preds[movie_indices] * 5,
                                'Autoencoder Prediction': ae_preds[movie_indices] * 5
                            })
                            
                            st.dataframe(comparison_df.style.format({
                                'Actual': '{:.1f}',
                                'RBM Prediction': '{:.1f}',
                                'Autoencoder Prediction': '{:.1f}'
                            }))
                            
                        except Exception as e:
                            st.error(f"Error generating comparison: {e}")
    
    with tab4:
        st.header("About This App")
        st.write("""
        This movie recommendation system compares two different neural network approaches:
        
        1. **Restricted Boltzmann Machine (RBM)**:
        - A generative stochastic neural network that can learn a probability distribution
        - Uses unsupervised learning to discover patterns in the data
        - Makes recommendations based on learned probability distributions
        
        2. **Autoencoder**:
        - An unsupervised neural network that learns efficient encodings of input data
        - Learns to compress and reconstruct user rating patterns
        - Makes recommendations based on reconstructed ratings
        
        ## How it works:
        1. The system is trained on the MovieLens 1M dataset containing 1 million ratings from 6,000 users on 4,000 movies.
        2. Both models learn patterns in user ratings to understand preferences.
        3. When you rate movies, the system predicts how you would rate other movies you haven't seen.
        4. Movies with the highest predicted ratings are recommended to you.
        
        ## Technologies used:
        - PyTorch for the neural network implementations
        - Streamlit for the user interface
        - Python for data processing
        
        This project demonstrates the application of different unsupervised learning techniques in recommendation systems.
        """)

if __name__ == "__main__":
    main()