# =============================================================================
# app.py — Streamlit Web Application for Movie Recommendation System
#
# Launch: streamlit run app.py
# =============================================================================

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_all_data
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeFilteringRecommender

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="🎬 MovieRec AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — professional dark cinema theme ────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e2f0;
  }
  .stApp {
    background: linear-gradient(135deg, #0a0a14 0%, #0f0f22 50%, #0a0a14 100%);
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1f 0%, #141428 100%);
    border-right: 1px solid #2a2a4a;
  }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #1a0533 0%, #0d1a33 50%, #0a1a2e 100%);
    border: 1px solid #3d2a6e;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(124,106,247,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 30% 70%, rgba(247,108,108,0.08) 0%, transparent 50%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c6af7 0%, #f76c6c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
  }
  .hero-sub {
    color: #8888aa;
    font-size: 1rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
  }

  /* ── Movie card ── */
  .movie-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(124,106,247,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s ease, background 0.2s ease;
  }
  .movie-card:hover {
    background: rgba(124,106,247,0.08);
    border-color: rgba(124,106,247,0.5);
  }
  .rank-badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #7c6af7;
    min-width: 2.2rem;
    text-align: center;
  }
  .rank-badge.gold   { color: #f7c948; }
  .rank-badge.silver { color: #b0b0c0; }
  .rank-badge.bronze { color: #cd7f32; }
  .movie-info { flex: 1; min-width: 0; }
  .movie-title-card {
    font-weight: 600;
    font-size: 0.95rem;
    color: #e8e8ff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .movie-meta {
    font-size: 0.78rem;
    color: #6666aa;
    margin-top: 0.2rem;
  }
  .score-pill {
    background: linear-gradient(135deg, rgba(124,106,247,0.2), rgba(247,108,108,0.15));
    border: 1px solid rgba(124,106,247,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #9d8fff;
    white-space: nowrap;
  }

  /* ── Section headers ── */
  .section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #7c6af7;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(124,106,247,0.4), transparent);
  }

  /* ── Method chip ── */
  .method-chip {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
  }
  .cb-chip  { background: rgba(124,106,247,0.15); color: #9d8fff; border: 1px solid rgba(124,106,247,0.3); }
  .cf-chip  { background: rgba(247,108,108,0.12); color: #ff9a9a; border: 1px solid rgba(247,108,108,0.3); }

  /* ── Info box ── */
  .info-box {
    background: rgba(124,106,247,0.07);
    border-left: 3px solid #7c6af7;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #aaaacc;
    margin-bottom: 1.2rem;
    line-height: 1.6;
  }

  /* ── Metric box ── */
  .metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }
  .metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c6af7, #f76c6c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .metric-label { font-size: 0.75rem; color: #6666aa; margin-top: 0.2rem; }

  /* ── Streamlit overrides ── */
  div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid #3d3d6d !important;
    border-radius: 8px !important;
    color: #e2e2f0 !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #7c6af7, #5c4dd7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }
  .stRadio > div > label { color: #aaaacc !important; font-size: 0.9rem !important; }
  div[data-testid="stSlider"] { color: #aaaacc !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached data & model loading ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading MovieLens dataset and training models…")
def load_models():
    """Load data and train both models once; cache for the session."""
    movies, ratings, _ = load_all_data()
    cb = ContentBasedRecommender()
    cb.fit(movies)
    cf = CollaborativeFilteringRecommender(min_ratings=20)
    cf.fit(movies, ratings)
    return movies, ratings, cb, cf


# ── Render a single recommendation card ────────────────────────────────────────
def render_movie_card(rank: int, title: str, genres: str, year, score: float):
    rank_class = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "")
    year_str = str(int(year)) if pd.notna(year) else "—"
    genres_str = genres if genres and genres != "(no genres listed)" else "—"
    score_pct = f"{score*100:.1f}%"

    st.markdown(f"""
    <div class="movie-card">
      <div class="rank-badge {rank_class}">#{rank}</div>
      <div class="movie-info">
        <div class="movie-title-card">{title}</div>
        <div class="movie-meta">🎭 {genres_str} &nbsp;·&nbsp; 📅 {year_str}</div>
      </div>
      <div class="score-pill">match {score_pct}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main app ────────────────────────────────────────────────────────────────────
def main():
    # ── Load models ─────────────────────────────────────────────────────────────
    movies, ratings, cb_model, cf_model = load_models()
    all_titles = sorted(movies["clean_title"].dropna().unique().tolist())

    # ── Hero banner ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">🎬 MovieRec AI</div>
      <div class="hero-sub">Content-Based &amp; Collaborative Filtering · MovieLens Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset stats row ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{len(movies):,}", "Movies"),
        (f"{len(ratings):,}", "Ratings"),
        (f"{ratings['userId'].nunique():,}", "Users"),
        (f"{movies['genres_list'].explode().nunique():,}", "Genres"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sidebar controls ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;
                    font-weight:700;color:#7c6af7;margin-bottom:1.2rem;">
          ⚙️ Settings
        </div>""", unsafe_allow_html=True)

        method = st.radio(
            "Recommendation Method",
            ["Both", "Content-Based Only", "Collaborative Only"],
            index=0
        )

        top_n = st.slider("Top N Recommendations", min_value=5, max_value=20, value=10, step=1)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.8rem;color:#6666aa;line-height:1.7;">
        <b style="color:#aaaacc;">Content-Based</b><br>
        Uses movie genres + TF-IDF to find similar films.<br><br>
        <b style="color:#aaaacc;">Collaborative</b><br>
        Uses rating patterns from 610 real users to surface films liked by similar audiences.
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;color:#555577;line-height:1.6;">
        🎓 B.Tech CSE Portfolio Project<br>
        Dataset: MovieLens 100K (GroupLens)<br>
        Stack: Python · Scikit-learn · Pandas · Streamlit
        </div>""", unsafe_allow_html=True)

    # ── Movie selector ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Pick a Movie</div>', unsafe_allow_html=True)

    default_idx = all_titles.index("Toy Story") if "Toy Story" in all_titles else 0
    selected_title = st.selectbox(
        "Search or select a movie title:",
        options=all_titles,
        index=default_idx,
        label_visibility="collapsed"
    )

    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        recommend_clicked = st.button("🎬 Get Recommendations")

    # ── Results ──────────────────────────────────────────────────────────────────
    if recommend_clicked and selected_title:
        st.markdown("<br>", unsafe_allow_html=True)

        use_cb = method in ("Both", "Content-Based Only")
        use_cf = method in ("Both", "Collaborative Only")

        if use_cb and use_cf:
            left, right = st.columns(2)
        else:
            left = right = None

        # ── Content-Based ────────────────────────────────────────────────────────
        if use_cb:
            target = left if left else st.container()
            with target:
                st.markdown('<span class="method-chip cb-chip">📚 Content-Based Filtering</span>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                  Recommends movies with <b>similar genres</b> to your selection
                  using TF-IDF + Cosine Similarity on the genre feature vector.
                </div>""", unsafe_allow_html=True)
                try:
                    cb_recs, matched = cb_model.recommend(selected_title, top_n=top_n)
                    st.markdown(f'<div style="font-size:0.8rem;color:#6666aa;margin-bottom:0.8rem;">Matched: <b style="color:#9d8fff;">{matched}</b></div>', unsafe_allow_html=True)
                    for _, row in cb_recs.iterrows():
                        render_movie_card(
                            int(row["rank"]), row["title"],
                            row.get("genres", ""), row.get("year"),
                            row["similarity_score"]
                        )
                except ValueError as e:
                    st.error(str(e))

        # ── Collaborative ────────────────────────────────────────────────────────
        if use_cf:
            target = right if right else st.container()
            with target:
                st.markdown('<span class="method-chip cf-chip">🤝 Collaborative Filtering</span>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                  Recommends movies <b>rated similarly</b> by real users, discovering
                  cross-genre patterns invisible to content features.
                </div>""", unsafe_allow_html=True)
                try:
                    cf_recs, matched = cf_model.recommend(selected_title, top_n=top_n)
                    st.markdown(f'<div style="font-size:0.8rem;color:#6666aa;margin-bottom:0.8rem;">Matched: <b style="color:#ff9a9a;">{matched}</b></div>', unsafe_allow_html=True)
                    for _, row in cf_recs.iterrows():
                        render_movie_card(
                            int(row["rank"]), row["title"],
                            "", None,
                            row["similarity_score"]
                        )
                except ValueError as e:
                    st.warning(str(e))

    # ── EDA section ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Explore Dataset Analysis (EDA)"):
        eda_path = os.path.join(os.path.dirname(__file__), "static", "eda_dashboard.png")
        if os.path.exists(eda_path):
            st.image(eda_path, use_container_width=True)
        else:
            st.info("Run `python main.py` first to generate the EDA dashboard.")

    # ── Theory section ───────────────────────────────────────────────────────────
    with st.expander("🧠 How it works — Theory"):
        st.markdown("""
        #### Content-Based Filtering
        Each movie is represented as a **TF-IDF vector** over its genres.
        
        **Cosine Similarity** between two movie vectors A and B:
        
        > cos(θ) = (A · B) / (|A| × |B|)
        
        A score of **1.0** = perfectly similar genres, **0.0** = no overlap.
        
        ---
        #### Collaborative Filtering (Item-Based)
        A **User × Movie rating matrix** is built. Each movie column becomes a vector
        of ratings across all users. Cosine similarity between two movie columns
        captures how similarly audiences rated them — independent of genres.
        
        **Advantage:** Can discover that a Horror fan also tends to enjoy Thrillers,
        even if the genres differ — because real users signal this through ratings.
        
        ---
        #### Key Difference
        | Dimension | Content-Based | Collaborative |
        |-----------|--------------|---------------|
        | Data used | Movie features (genres) | User ratings |
        | Cold start | ✅ Works for new users | ❌ Needs rating history |
        | Serendipity | ❌ Stays within genres | ✅ Cross-genre discovery |
        | Scalability | ✅ Very fast | ⚠️ Matrix grows with users |
        """)


if __name__ == "__main__":
    main()
