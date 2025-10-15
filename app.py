import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import re
from urllib.parse import quote
from difflib import get_close_matches


OMDB_API_KEY = "2a400fd6"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/300x450?text=No+Image"

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎥 Smart Movie Recommender System")


def normalize(text):
    """Normalize text for better matching"""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def fetch_poster(title):
    clean_title = title.split('(')[0].strip()
    url = f"http://www.omdbapi.com/?t={quote(clean_title)}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5).json()
        poster = response.get('Poster')
        if poster and poster != "N/A":
            return poster
        else:
            return PLACEHOLDER_IMAGE
    except:
        return PLACEHOLDER_IMAGE



try:
    movies = pickle.load(open('model/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('model/similarity.pkl', 'rb'))
    similarity = np.array(similarity)
except:
    st.error("Could not load pickle files. Make sure they exist in the 'model' folder.")
    st.stop()



def recommend(search_term, top_n=5):
    try:
        # Normalize columns
        movies['norm_title'] = movies['title'].apply(normalize)
        movies['norm_genres'] = movies['genres'].apply(normalize)
        movies['norm_cast'] = movies['cast'].apply(normalize)

        search_norm = normalize(search_term)

        mask = (
            movies['norm_title'].str.contains(search_norm, na=False) |
            movies['norm_genres'].str.contains(search_norm, na=False) |
            movies['norm_cast'].str.contains(search_norm, na=False)
        )
        matched = movies[mask]

        if matched.empty:
            all_combined = (movies['title'] + ' ' + movies['genres'] + ' ' + movies['cast']).str.lower()
            closest = get_close_matches(search_term.lower(), all_combined.tolist(), n=1, cutoff=0.3)
            if closest:
                idx = all_combined[all_combined == closest[0]].index[0]
                distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
                matched = movies.iloc[[i[0] for i in distances[1:top_n+1]]]
            else:
                st.warning("No similar movies found 😔")
                return [], []

        recommended_movie_names = matched.head(top_n)['title'].tolist()
        recommended_movie_posters = [fetch_poster(t) for t in recommended_movie_names]

        return recommended_movie_names, recommended_movie_posters
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return [], []
    
all_movies = movies['title'].tolist()




search_input = st.text_input(
    "🔍 Search by movie, genre, or cast",
    placeholder="e.g. Action, Spider Man, Robert Downey Jr."
)


if st.button("Show Recommendations"):
    if not search_input.strip():
        st.warning("Please enter a search term.")
    else:
        recommended_movie_names, recommended_movie_posters = recommend(search_input, top_n=5)
        if recommended_movie_names:
            cols = st.columns(len(recommended_movie_names))
            for idx, col in enumerate(cols):
                with col:
                    st.text(recommended_movie_names[idx])
                    st.image(recommended_movie_posters[idx], use_container_width=True)
        else:
            st.warning("No recommendations found.")