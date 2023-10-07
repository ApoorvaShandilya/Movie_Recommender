import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def gd_path(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def load_data():
    # Load the data files
    files_id = {
        'links': "1GR8IQ2OXsFI8MNmv4bQIV1XXkq7n56MB",
        'movies': "1PDuCaAhhVTRLYdftMr6VqX23crMqB_qg",
        'rating': "1F4_-HBPBSySMjxdGxlykWVjvVn9AJ0BS",
        'tags': "1bH6HhZfqLT0JGqYxyRLQAk7UIpnYj4x4"
    }

    links = pd.read_csv(gd_path(files_id['links']), sep=",")
    movies = pd.read_csv(gd_path(files_id['movies']), sep=",")
    rating = pd.read_csv(gd_path(files_id['rating']), sep=",")
    tags = pd.read_csv(gd_path(files_id['tags']), sep=",")

    return links, movies, rating, tags

def preprocess_data(rating):
    # Create a user-item matrix
    movie_item = pd.pivot_table(data=rating,
                                values='rating',
                                index='userId',
                                columns='movieId')
    movie_item.fillna(0, inplace=True)
    return movie_item

def train_recommender_model(movie_item):
    # Compute cosine similarities
    movie_similarities = pd.DataFrame(cosine_similarity(movie_item),
                                      columns=movie_item.index,
                                      index=movie_item.index)
    return movie_similarities

def recommend_movies(user_id, movie_item, movie_similarities, movie_name, n=5):
    # Compute the weights for the specified user
    weights = (
        movie_similarities.query("userId!=@user_id")[user_id] /
        sum(movie_similarities.query("userId!=@user_id")[user_id])
    )

    # Find movies not rated by the user
    movies_not_rated_by_user = movie_item.loc[user_id] == 0

    # Select movies that the inputed user has not rated
    not_rated_movies = movie_item.loc[movie_item.index!=user_id, movies_not_rated_by_user]

    # Compute the ratings user would give to those unrated movies
    weighted_averages = pd.DataFrame(not_rated_movies.T.dot(weights), columns=["predicted_rating"])

    # Merge with movie names
    recommendations = weighted_averages.merge(movie_name, left_index=True, right_on="movieId")

    # Sort recommendations by predicted rating in descending order
    top_recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(n)

    return top_recommendations

def main():
    st.title("Movie Recommender System")

    # Load the data
    links, movies, rating, tags = load_data()
    movie_item = preprocess_data(rating)
    movie_similarities = train_recommender_model(movie_item)
    movie_name = movies[['movieId', 'title']]

    # User input
    user_id = st.number_input("Enter User ID", min_value=1, max_value=1000)
    n_recommendations = st.slider("Number of Recommendations", 1, 20, 5)

    if st.button("Get Recommendations"):
        recommendations = recommend_movies(user_id, movie_item, movie_similarities, movie_name, n=n_recommendations)
        st.subheader("Top Movie Recommendations:")
        st.table(recommendations)

if __name__ == "__main__":
    main()