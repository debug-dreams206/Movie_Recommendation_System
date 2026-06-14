# =============================================================================
# models/content_based.py
# Purpose: Content-Based Filtering using TF-IDF and Cosine Similarity
#
# THEORY:
# -------
# Content-Based Filtering recommends items SIMILAR TO what a user has liked
# before. It uses ITEM FEATURES (genres, tags, descriptions) rather than
# other users' behaviour.
#
# How it works here:
#  1. Represent each movie as a "bag of genres" (TF-IDF weighted vector)
#  2. Compute Cosine Similarity between all movie vectors
#  3. Given a query movie, return the N most similar movies
#
# Cosine Similarity: cos(θ) = (A · B) / (|A| × |B|)
#   → 1.0  = identical direction (very similar)
#   → 0.0  = perpendicular (no shared genres)
#   → -1.0 = opposite direction (completely different)
#
# Pros: Works without any user data (cold-start friendly for NEW users)
# Cons: Limited to features we give it; can't discover serendipitous picks
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Movie recommender based purely on movie content features (genres).
    
    Attributes:
      movies        : the movies DataFrame
      tfidf_matrix  : TF-IDF matrix (movies × genre_terms)
      cosine_sim    : precomputed cosine similarity matrix (movies × movies)
      indices       : mapping from clean_title → DataFrame index
    """

    def __init__(self):
        self.movies = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    # ------------------------------------------------------------------
    # Step 1: Build TF-IDF matrix and cosine similarity matrix
    # ------------------------------------------------------------------
    def fit(self, movies: pd.DataFrame):
        """
        Train the content-based model.
        
        Steps:
          a) Convert genre lists to space-separated strings (TF-IDF input)
          b) Apply TF-IDF vectorization
          c) Compute pairwise cosine similarity
          d) Build a title → index lookup table
        
        Args:
          movies (pd.DataFrame): cleaned movies DataFrame from data_loader
        """
        self.movies = movies.reset_index(drop=True)

        # a) Genre string: ["Action", "Adventure"] → "Action Adventure"
        #    Using space separator makes each genre a separate TF-IDF term
        self.movies["genre_str"] = self.movies["genres_list"].apply(
            lambda lst: " ".join(lst) if lst else "unknown"
        )

        # b) TF-IDF Vectorizer
        #    - analyzer='word'  : tokenize on whitespace
        #    - ngram_range=(1,2): also capture bigrams like "Science Fiction"
        #    - min_df=1         : include terms appearing in at least 1 doc
        print("🔧 Building TF-IDF matrix...")
        tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = tfidf.fit_transform(self.movies["genre_str"])
        print(f"   TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        # → (n_movies, n_unique_genre_terms)

        # c) Compute cosine similarity between every pair of movie vectors
        #    Result is an (n_movies × n_movies) dense matrix.
        #    cosine_sim[i][j] = similarity between movie i and movie j
        print("🔧 Computing cosine similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print(f"   Similarity matrix shape: {self.cosine_sim.shape}")

        # d) Title → row index lookup (use clean_title, lowercase for matching)
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies["clean_title"].str.lower()
        )
        print("✅ Content-Based model ready!\n")

    # ------------------------------------------------------------------
    # Step 2: Recommend movies similar to a given title
    # ------------------------------------------------------------------
    def recommend(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return the top-N most similar movies to the given title.
        
        Args:
          movie_title (str): exact or partial movie title (case-insensitive)
          top_n       (int): how many recommendations to return
        
        Returns:
          pd.DataFrame with columns: rank, title, genres, year, similarity_score
        
        Raises:
          ValueError: if movie title not found in the dataset
        """
        title_lower = movie_title.lower().strip()

        # Fuzzy match: find the best title containing the search term
        matches = [t for t in self.indices.index if title_lower in t]
        if not matches:
            raise ValueError(
                f"❌ Movie '{movie_title}' not found. "
                "Try a different title or check spelling."
            )

        # If multiple matches, pick the first (closest)
        best_match = matches[0]
        idx = self.indices[best_match]

        # Handle duplicate titles (take the first one)
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Get similarity scores for this movie vs. all others
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort by similarity descending, skip index 0 (the movie itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # skip self

        # Extract movie indices and scores
        movie_indices = [s[0] for s in sim_scores]
        scores = [round(s[1], 4) for s in sim_scores]

        # Build result DataFrame
        results = self.movies.iloc[movie_indices][
            ["clean_title", "genres", "year"]
        ].copy()
        results["similarity_score"] = scores
        results.insert(0, "rank", range(1, len(results) + 1))
        results = results.rename(columns={"clean_title": "title"})
        results = results.reset_index(drop=True)

        return results, best_match

    # ------------------------------------------------------------------
    # Utility: Get all available movie titles (for UI dropdowns)
    # ------------------------------------------------------------------
    def get_all_titles(self):
        """Return list of all clean movie titles in the dataset."""
        return self.movies["clean_title"].sort_values().tolist()
