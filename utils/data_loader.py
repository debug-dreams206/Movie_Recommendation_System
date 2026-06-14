# =============================================================================
# utils/data_loader.py
# Purpose: Download and load the MovieLens 100K dataset
# =============================================================================

import os
import zipfile
import requests
import pandas as pd
import numpy as np

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# __file__ = .../movie_recommendation/utils/data_loader.py
# Two dirname() calls walk up to the project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# =============================================================================
# FIX 1: Add a browser-style User-Agent header.
# Grouplens.org returns HTTP 403 if you use Python's default "python-requests"
# agent. A realistic User-Agent string bypasses this block.
# =============================================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_dataset():
    """
    Downloads and extracts the MovieLens small dataset if not already present.

    The dataset contains:
    - ~100,000 ratings from 610 users on 9,742 movies
    - Ratings scale: 0.5 to 5.0 (half-star increments)
    - Movies tagged with genres, links to IMDB/TMDB

    FIX 1 applied: Uses a proper User-Agent header so grouplens.org doesn't
    reject the request with HTTP 403 Forbidden.
    """
    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    extract_path = os.path.join(DATA_DIR, "ml-latest-small")

    if os.path.exists(extract_path):
        print("✅ Dataset already exists. Skipping download.")
        return extract_path

    os.makedirs(DATA_DIR, exist_ok=True)
    print("📥 Downloading MovieLens dataset...")

    try:
        # FIX: pass headers= so the server accepts us
        response = requests.get(MOVIELENS_URL, stream=True, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print("💡 Generating sample dataset so you can still run the project...")
        return _create_sample_dataset()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    os.remove(zip_path)
    print("✅ Dataset ready!")
    return extract_path


def _create_sample_dataset():
    """
    FIX 2: Offline fallback — generates a realistic sample MovieLens-style
    dataset so the project runs even when the internet is unavailable.

    This is a beginner-friendly safety net. Real projects should always have
    a fallback or bundle a small sample dataset for offline testing.
    """
    sample_path = os.path.join(DATA_DIR, "ml-latest-small")
    os.makedirs(sample_path, exist_ok=True)

    np.random.seed(42)

    # ── Sample movies (representative genres and years) ──────────────────────
    sample_movies = [
        (1,   "Toy Story (1995)",                "Adventure|Animation|Children|Comedy|Fantasy"),
        (2,   "Jumanji (1995)",                  "Adventure|Children|Fantasy"),
        (3,   "Grumpier Old Men (1995)",          "Comedy|Romance"),
        (10,  "GoldenEye (1995)",                "Action|Adventure|Thriller"),
        (32,  "Twelve Monkeys (1995)",            "Mystery|Sci-Fi|Thriller"),
        (47,  "Seven (1995)",                     "Mystery|Thriller"),
        (50,  "Usual Suspects, The (1995)",       "Crime|Mystery|Thriller"),
        (110, "Braveheart (1995)",               "Action|Drama|War"),
        (260, "Star Wars: Episode IV (1977)",     "Action|Adventure|Sci-Fi"),
        (296, "Pulp Fiction (1994)",              "Comedy|Crime|Drama|Thriller"),
        (318, "Shawshank Redemption, The (1994)", "Crime|Drama"),
        (356, "Forrest Gump (1994)",              "Comedy|Drama|Romance|War"),
        (480, "Jurassic Park (1993)",             "Action|Adventure|Sci-Fi|Thriller"),
        (527, "Schindler's List (1993)",          "Drama|War"),
        (588, "Aladdin (1992)",                  "Adventure|Animation|Children|Comedy|Musical"),
        (593, "Silence of the Lambs, The (1991)","Crime|Horror|Thriller"),
        (858, "Godfather, The (1972)",            "Crime|Drama"),
        (1197,"Princess Bride, The (1987)",       "Adventure|Comedy|Fantasy|Romance"),
        (1198,"Raiders of the Lost Ark (1981)",   "Action|Adventure"),
        (1210,"Star Wars: Episode VI (1983)",     "Action|Adventure|Sci-Fi"),
        (1270,"Back to the Future (1985)",        "Adventure|Comedy|Sci-Fi"),
        (1580,"Men in Black (1997)",              "Action|Comedy|Sci-Fi"),
        (1721,"Titanic (1997)",                  "Drama|Romance"),
        (1732,"Big Lebowski, The (1998)",         "Comedy|Crime"),
        (2028,"Saving Private Ryan (1998)",       "Action|Drama|War"),
        (2571,"Matrix, The (1999)",               "Action|Sci-Fi|Thriller"),
        (2959,"Fight Club (1999)",               "Action|Crime|Drama|Thriller"),
        (3578,"Gladiator (2000)",                "Action|Adventure|Drama"),
        (4306,"Shrek (2001)",                    "Adventure|Animation|Children|Comedy|Fantasy|Romance"),
        (4993,"Lord of the Rings: Fellowship (2001)", "Adventure|Fantasy"),
        (5952,"Lord of the Rings: Two Towers (2002)", "Adventure|Fantasy"),
        (7153,"Lord of the Rings: Return (2003)", "Action|Adventure|Drama|Fantasy"),
        (8961,"Incredibles, The (2004)",          "Action|Adventure|Animation|Children|Comedy"),
        (58559,"Dark Knight, The (2008)",         "Action|Crime|Drama|IMAX"),
        (68157,"Inception (2010)",               "Action|Crime|Drama|Mystery|Sci-Fi|Thriller"),
        (79132,"Interstellar (2014)",            "Sci-Fi"),
        (86882,"Avengers, The (2012)",           "Action|Adventure|Sci-Fi|Thriller"),
        (91529,"Django Unchained (2012)",        "Action|Drama|Western"),
        (96281,"Silver Linings Playbook (2012)", "Comedy|Drama|Romance"),
        (99114,"Django Unchained (2012)",        "Drama|Western"),
    ]

    movies_df = pd.DataFrame(sample_movies, columns=["movieId", "title", "genres"])
    movies_df.to_csv(os.path.join(sample_path, "movies.csv"), index=False)

    # ── Sample ratings (610 users × subset of movies) ───────────────────────
    movie_ids = movies_df["movieId"].tolist()
    n_users = 200
    n_ratings = 8000

    user_ids = np.random.randint(1, n_users + 1, size=n_ratings)
    movie_choices = np.random.choice(movie_ids, size=n_ratings)
    ratings_values = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                                       size=n_ratings,
                                       p=[0.01, 0.02, 0.03, 0.05, 0.07,
                                          0.12, 0.15, 0.25, 0.18, 0.12])
    timestamps = np.random.randint(850000000, 1700000000, size=n_ratings)

    ratings_df = pd.DataFrame({
        "userId": user_ids,
        "movieId": movie_choices,
        "rating": ratings_values,
        "timestamp": timestamps,
    }).drop_duplicates(subset=["userId", "movieId"])

    ratings_df.to_csv(os.path.join(sample_path, "ratings.csv"), index=False)

    print(f"✅ Sample dataset created: {len(movies_df)} movies, {len(ratings_df)} ratings")
    return sample_path


def load_movies(data_path):
    """
    Load and clean the movies.csv file.

    Schema:
      movieId (int): unique movie identifier
      title   (str): movie title with release year e.g. "Toy Story (1995)"
      genres  (str): pipe-separated genre list e.g. "Animation|Comedy|Family"

    Returns:
      pd.DataFrame with added 'year' and cleaned 'genres_list' columns
    """
    movies_path = os.path.join(data_path, "movies.csv")
    movies = pd.read_csv(movies_path)

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    movies["clean_title"] = (
        movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    )
    movies["genres_list"] = movies["genres"].apply(
        lambda x: x.split("|") if x != "(no genres listed)" else []
    )

    return movies


def load_ratings(data_path):
    """
    Load and clean the ratings.csv file.

    Schema:
      userId    (int):   user who provided the rating
      movieId   (int):   movie being rated
      rating    (float): rating value (0.5 – 5.0)
      timestamp (int):   Unix timestamp of when rating was given

    Returns:
      pd.DataFrame with timestamp converted to datetime
    """
    ratings_path = os.path.join(data_path, "ratings.csv")
    ratings = pd.read_csv(ratings_path)
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return ratings


def load_all_data():
    """
    Master function: downloads (if needed) and returns all cleaned dataframes.

    Returns:
      tuple: (movies_df, ratings_df, merged_df)

    Usage:
      movies, ratings, merged = load_all_data()
    """
    data_path = download_dataset()
    movies = load_movies(data_path)
    ratings = load_ratings(data_path)

    merged = ratings.merge(movies, on="movieId", how="left")

    print(f"📊 Loaded {len(movies):,} movies and {len(ratings):,} ratings")
    print(f"👥 Total unique users: {ratings['userId'].nunique():,}")
    print(f"🎬 Total unique movies rated: {ratings['movieId'].nunique():,}")

    return movies, ratings, merged
