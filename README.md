# 🎬 Movie Recommendation System using ML

This is a simple **Content-Based Movie Recommender** built using **TF-IDF vectorization** and **cosine similarity**. It takes your favorite movie as input and suggests a list of similar movies based on key features like genre, cast, keywords, tagline, and director.

---

## Features

-  **Content-Based Filtering**: Recommends movies similar in theme and content.
-  **TF-IDF + Cosine Similarity**: Transforms text features into vectors and finds closeness between movies.
-  **Fuzzy Matching**: Accepts user input even with spelling errors using Python's `difflib`.

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Difflib
- Google Colab (for development)

---

## 📂 Dataset

- `movies.csv` 
  - `index`
  - `title`
  - `genres`
  - `keywords`
  - `tagline`
  - `cast`
  - `director`

Dataset source: [Kaggle - TMDB Movie Dataset](https://www.kaggle.com/datasets)

---

## 💻 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjayPatendla082/movie-recommendation-using-ml.git
   cd movie-recommender
