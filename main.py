#!/usr/bin/env python3
# =============================================================================
# main.py — Movie Recommendation System
# Author  : Your Name
# Project : MovieLens Recommendation Engine (B.Tech CSE Portfolio Project)
#
# This script demonstrates the full ML pipeline:
#   1. Load & clean the MovieLens dataset
#   2. Exploratory Data Analysis (EDA)
#   3. Content-Based Filtering  (TF-IDF + Cosine Similarity on genres)
#   4. Collaborative Filtering  (Item-Based CF + Cosine Similarity on ratings)
#   5. Side-by-side recommendation comparison
#   6. Model evaluation (precision proxy via rating cross-check)
#
# Run: python main.py
# =============================================================================

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress sklearn/scipy warnings for clean output
warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_loader import load_all_data
from utils.eda import print_summary, plot_eda
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender

# ── ASCII art banner ───────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║        🎬  MOVIE RECOMMENDATION SYSTEM  🎬                      ║
║        Built with MovieLens • TF-IDF • Cosine Similarity        ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ==============================================================================
# Section 1 — Display formatted recommendation table
# ==============================================================================
def display_recommendations(results: pd.DataFrame, method: str, query: str):
    """
    Pretty-print a recommendation DataFrame as a formatted ASCII table.
    
    Args:
      results (pd.DataFrame): output from recommend() functions
      method  (str)         : "Content-Based" or "Collaborative Filtering"
      query   (str)         : the movie title searched for
    """
    line = "─" * 70
    print(f"\n{line}")
    print(f"  🎯  {method.upper()} RECOMMENDATIONS")
    print(f"  Based on: \"{query}\"")
    print(line)
    print(f"  {'#':<4}  {'MOVIE TITLE':<40}  {'YEAR':<6}  {'SCORE':>6}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*6}  {'-'*6}")

    for _, row in results.iterrows():
        title = str(row.get("title", "N/A"))
        title_trunc = (title[:37] + "...") if len(title) > 40 else title
        year = str(int(row["year"])) if "year" in row and pd.notna(row.get("year")) else "—"
        score = f"{row['similarity_score']:.4f}"
        print(f"  {int(row['rank']):<4}  {title_trunc:<40}  {year:<6}  {score:>6}")

    print(line)


# ==============================================================================
# Section 2 — Simple CLI interface for interactive use
# ==============================================================================
def interactive_cli(cb_model: ContentBasedRecommender,
                    cf_model: CollaborativeFilteringRecommender):
    """
    Let the user type a movie name and receive recommendations interactively.
    Press Ctrl-C or type 'quit' to exit.
    """
    print("\n" + "="*60)
    print("  🎮  INTERACTIVE RECOMMENDATION CONSOLE")
    print("  Type a movie title to get recommendations.")
    print("  Type 'quit' to exit.")
    print("="*60)

    while True:
        try:
            query = input("\n🎬 Enter movie title: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break
            if not query:
                continue

            # Content-Based
            try:
                cb_recs, matched = cb_model.recommend(query, top_n=10)
                display_recommendations(cb_recs, "Content-Based Filtering", matched)
            except ValueError as e:
                print(f"  ⚠️  Content-Based: {e}")

            # Collaborative
            try:
                cf_recs, matched = cf_model.recommend(query, top_n=10)
                display_recommendations(cf_recs, "Collaborative Filtering", matched)
            except ValueError as e:
                print(f"  ⚠️  Collaborative: {e}")

        except KeyboardInterrupt:
            print("\n\n👋 Exiting. Goodbye!")
            break


# ==============================================================================
# Section 3 — Evaluation: Precision Proxy
# ==============================================================================
def evaluate_models(movies, ratings, cb_model, cf_model):
    """
    Evaluate recommendation quality using a simple precision proxy:
    For the top-5 most-rated movies, check what % of recommendations
    have above-average ratings (>= 3.5 stars).
    
    Note: For a rigorous evaluation you'd use train/test split and compute
    NDCG, MAP, or RMSE. This proxy gives a quick sanity check.
    """
    print("\n" + "="*60)
    print("  📏  MODEL EVALUATION (Precision Proxy @ N=10)")
    print("="*60)

    # Average rating per movie
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    good_threshold = 3.5

    # Pick 5 popular test movies
    popular_ids = (ratings.groupby("movieId").size()
                   .sort_values(ascending=False)
                   .head(20).index.tolist())

    test_titles = []
    for mid in popular_ids[:5]:
        title = movies.loc[movies["movieId"] == mid, "clean_title"].values
        if len(title) > 0:
            test_titles.append(title[0])

    print(f"\n  Test movies: {test_titles}\n")
    print(f"  {'Movie':<30}  {'CB Precision':>14}  {'CF Precision':>14}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*14}")

    for title in test_titles:
        cb_prec = cf_prec = "N/A"

        try:
            cb_recs, _ = cb_model.recommend(title, top_n=10)
            # For each recommendation, check if its avg rating >= threshold
            cb_movie_ids = movies.loc[
                movies["clean_title"].isin(cb_recs["title"]), "movieId"
            ].tolist()
            cb_good = sum(1 for mid in cb_movie_ids
                         if mid in avg_ratings.index and avg_ratings[mid] >= good_threshold)
            cb_prec = f"{cb_good}/{min(len(cb_movie_ids),10)} ({100*cb_good//max(len(cb_movie_ids),1)}%)"
        except Exception:
            pass

        try:
            cf_recs, _ = cf_model.recommend(title, top_n=10)
            cf_movie_ids = movies.loc[
                movies["clean_title"].isin(cf_recs["title"]), "movieId"
            ].tolist()
            cf_good = sum(1 for mid in cf_movie_ids
                         if mid in avg_ratings.index and avg_ratings[mid] >= good_threshold)
            cf_prec = f"{cf_good}/{min(len(cf_movie_ids),10)} ({100*cf_good//max(len(cf_movie_ids),1)}%)"
        except Exception:
            pass

        title_short = (title[:27] + "...") if len(title) > 30 else title
        print(f"  {title_short:<30}  {cb_prec:>14}  {cf_prec:>14}")

    print("\n  (Precision = % of recommended movies with avg rating ≥ 3.5)")
    print("="*60 + "\n")


# ==============================================================================
# MAIN — orchestrate the full pipeline
# ==============================================================================
def main():
    print(BANNER)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("━"*60)
    print("  STEP 1: Loading Dataset")
    print("━"*60)
    movies, ratings, merged = load_all_data()

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    print("\n━"*60)
    print("  STEP 2: Exploratory Data Analysis")
    print("━"*60)
    print_summary(movies, ratings)
    eda_path = plot_eda(movies, ratings)

    # ── 3. Train Content-Based Model ──────────────────────────────────────────
    print("\n━"*60)
    print("  STEP 3: Training Content-Based Filtering Model")
    print("━"*60)
    print("""
  📚 THEORY — Content-Based Filtering
  ─────────────────────────────────────
  Represents each movie as a TF-IDF vector of its genres.
  Cosine Similarity measures the angle between two vectors:

    cos(θ) = (A · B) / (|A| × |B|)

  A score of 1.0 = identical genres, 0.0 = no genre overlap.
  """)
    cb_model = ContentBasedRecommender()
    cb_model.fit(movies)

    # ── 4. Train Collaborative Model ──────────────────────────────────────────
    print("━"*60)
    print("  STEP 4: Training Collaborative Filtering Model")
    print("━"*60)
    print("""
  📚 THEORY — Collaborative Filtering (Item-Based)
  ─────────────────────────────────────────────────
  Builds a User × Movie rating matrix. Movies are similar
  if they were rated similarly by the same users.

  Item similarity = cosine_similarity(rating_vector_A, rating_vector_B)

  This captures taste that goes beyond genre labels.
  """)
    cf_model = CollaborativeFilteringRecommender(min_ratings=20)
    cf_model.fit(movies, ratings)

    # ── 5. Demo Recommendations ───────────────────────────────────────────────
    print("━"*60)
    print("  STEP 5: Sample Recommendations")
    print("━"*60)

    demo_movies = ["Toy Story", "The Dark Knight", "Inception", "The Matrix"]

    for demo in demo_movies:
        # Content-Based
        try:
            cb_recs, matched = cb_model.recommend(demo, top_n=10)
            display_recommendations(cb_recs, "Content-Based Filtering", matched)
        except ValueError as e:
            print(e)

        # Collaborative
        try:
            cf_recs, matched = cf_model.recommend(demo, top_n=10)
            display_recommendations(cf_recs, "Collaborative Filtering", matched)
        except ValueError as e:
            print(e)

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print("━"*60)
    print("  STEP 6: Model Evaluation")
    print("━"*60)
    evaluate_models(movies, ratings, cb_model, cf_model)

    # ── 7. Interactive CLI (optional, comment out for non-interactive runs) ────
    print("━"*60)
    print("  STEP 7: Interactive Mode")
    print("━"*60)
    run_interactive = input("  Launch interactive CLI? (y/n): ").strip().lower()
    if run_interactive == "y":
        interactive_cli(cb_model, cf_model)
    else:
        print("\n  ⏩  Skipping interactive mode.")
        print("  💡  Run the Streamlit app: streamlit run app.py\n")

    print("""
╔══════════════════════════════════════════════════════════╗
║  ✅  Pipeline Complete!                                  ║
║  📊  EDA chart saved → static/eda_dashboard.png         ║
║  🌐  Launch web app  → streamlit run app.py             ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
