# Importing necessary libraries
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
movies_data = pd.read_csv('/content/movies.csv')  # Replace with your file path

# Display the first few rows of the dataset
print(movies_data.head())

# Display the shape of the dataset (rows, columns)
print(movies_data.shape)

# Select important features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print("Selected Features:", selected_features)

# Fill missing values with empty strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single string
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
print("Combined Features Sample:\n", combined_features.head())

# Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity between movies
similarity = cosine_similarity(feature_vectors)
print("Cosine Similarity Shape:", similarity.shape)

# Get movie name input from user
movie_name = input("Enter your favourite movie name: ")

# Create a list of all movie titles
list_of_all_titles = movies_data['title'].tolist()

# Find the closest matching title
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
if not find_close_match:
    print("No close match found. Please check the spelling.")
    exit()

close_match = find_close_match[0]
print("Did you mean:", close_match)

# Find the index of the matched movie
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

# Get a list of similar movies in descending order of similarity score
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# Print top 10 recommended movies
print("\nMovies suggested for you:\n")
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i < 11:
        print(i, ".", title_from_index)
        i += 1
