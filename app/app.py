import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Analytics",
    page_icon="🌍",
    layout="wide"
)

# ── FILE PATHS ─────────────────────────────────────────────────────────────────
DATA_PATH   = "data/cleaned/master_dataset.csv"
MODELS_PATH = "models"

# ── LOAD DATA & MODELS ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    reg_model   = joblib.load(os.path.join(MODELS_PATH, "regression_model.pkl"))
    clf_model   = joblib.load(os.path.join(MODELS_PATH, "classification_model.pkl"))
    le          = joblib.load(os.path.join(MODELS_PATH, "label_encoder.pkl"))
    user_matrix = joblib.load(os.path.join(MODELS_PATH, "user_item_matrix.pkl"))
    att_sim     = joblib.load(os.path.join(MODELS_PATH, "attraction_similarity.pkl"))
    scaler      = joblib.load(os.path.join(MODELS_PATH, "rec_scaler.pkl"))
    return reg_model, clf_model, le, user_matrix, att_sim, scaler

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

try:
    reg_model, clf_model, le, user_matrix, att_sim, scaler = load_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.title("🌍 Tourism Analytics")
page = st.sidebar.selectbox("Navigate", [
    "🏠 Home",
    "📊 EDA Dashboard",
    "⭐ Predict Rating",
    "🧳 Predict Visit Mode",
    "🎯 Get Recommendations"
])

# ── HOME ───────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🌍 Tourism Experience Analytics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Visits",  f"{len(df):,}")
    col2.metric("Unique Users",  f"{df['UserId'].nunique():,}")
    col3.metric("Attractions",   f"{df['AttractionId'].nunique():,}")
    col4.metric("Avg Rating",    f"{df['Rating'].mean():.2f}")

    st.markdown("---")
    st.markdown("""
    ### What this app does:
    - 📊 **EDA Dashboard** — Explore tourism trends and patterns
    - ⭐ **Predict Rating** — Predict how a user will rate an attraction
    - 🧳 **Predict Visit Mode** — Predict how a user will travel
    - 🎯 **Get Recommendations** — Get personalized attraction suggestions
    """)

# ── EDA DASHBOARD ──────────────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Ratings", "Visits", "Geography"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots()
            df['Rating'].value_counts().sort_index().plot(
                kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Season vs Rating")
            fig, ax = plt.subplots()
            sns.boxplot(x='Season', y='Rating', data=df, ax=ax,
                        hue='Season', legend=False, palette='Set3',
                        order=['Spring', 'Summer', 'Autumn', 'Winter'])
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Visit Mode Distribution")
            fig, ax = plt.subplots()
            vc = df['VisitMode'].value_counts()
            ax.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Top 10 Attractions")
            fig, ax = plt.subplots(figsize=(8, 5))
            top = df['Attraction'].value_counts().head(10)
            sns.barplot(x=top.values, y=top.index,
                        hue=top.index, legend=False, palette='viridis', ax=ax)
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Visits by Continent")
            fig, ax = plt.subplots()
            vc = df['Continent'].value_counts()
            sns.barplot(x=vc.values, y=vc.index,
                        hue=vc.index, legend=False, palette='Set1', ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Top 10 Countries")
            fig, ax = plt.subplots(figsize=(8, 5))
            top = df['Country'].value_counts().head(10)
            sns.barplot(x=top.values, y=top.index,
                        hue=top.index, legend=False, palette='coolwarm', ax=ax)
            st.pyplot(fig)
            plt.close(fig)

# ── PREDICT RATING ─────────────────────────────────────────────────────────────
elif page == "⭐ Predict Rating":
    st.title("⭐ Predict Attraction Rating")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        visit_year      = st.selectbox("Visit Year", sorted(df['VisitYear'].unique(), reverse=True))
        visit_month     = st.slider("Visit Month", 1, 12, 6)
        attraction_type = st.selectbox("Attraction Type", sorted(df['AttractionType'].unique()))
        continent       = st.selectbox("Continent", sorted(df['Continent'].unique()))

    with col2:
        region           = st.selectbox("Region", sorted(df['Region'].unique()))
        country          = st.selectbox("Country", sorted(df['Country'].unique()))
        user_avg_rating  = st.slider("Your Average Rating", 1.0, 5.0, 4.0, 0.1)
        user_visit_count = st.number_input("Your Visit Count", min_value=1, value=1)

    att_type_id  = df[df['AttractionType'] == attraction_type]['AttractionTypeId'].iloc[0]
    continent_id = df[df['Continent'] == continent]['ContinentId'].iloc[0]
    region_id    = df[df['Region'] == region]['RegionId'].iloc[0]
    country_id   = df[df['Country'] == country]['CountryId'].iloc[0]
    att_avg      = df['Attraction_Avg_Rating'].mean()
    att_cnt      = df['Attraction_Visit_Count'].mean()

    if st.button("⭐ Predict Rating", use_container_width=True):
        input_data = np.array([[
            visit_year, visit_month, att_type_id,
            continent_id, region_id, country_id,
            user_avg_rating, user_visit_count,
            att_avg, att_cnt
        ]])
        prediction = reg_model.predict(input_data)[0]
        prediction = round(min(max(prediction, 1), 5), 2)
        st.markdown("---")
        st.success(f"### Predicted Rating: {'⭐' * round(prediction)}  {prediction} / 5")

# ── PREDICT VISIT MODE ─────────────────────────────────────────────────────────
elif page == "🧳 Predict Visit Mode":
    st.title("🧳 Predict Visit Mode")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        visit_year  = st.selectbox("Visit Year", sorted(df['VisitYear'].unique(), reverse=True))
        visit_month = st.slider("Visit Month", 1, 12, 6)
        attraction  = st.selectbox("Attraction", sorted(df['Attraction'].unique()))
        continent   = st.selectbox("Continent", sorted(df['Continent'].unique()))

    with col2:
        region           = st.selectbox("Region", sorted(df['Region'].unique()))
        country          = st.selectbox("Country", sorted(df['Country'].unique()))
        rating           = st.slider("Your Rating", 1, 5, 4)
        user_avg_rating  = st.slider("Your Average Rating", 1.0, 5.0, 4.0, 0.1)
        user_visit_count = st.number_input("Your Visit Count", min_value=1, value=1)

    attraction_id   = df[df['Attraction'] == attraction]['AttractionId'].iloc[0]
    att_type_id     = df[df['Attraction'] == attraction]['AttractionTypeId'].iloc[0]
    continent_id    = df[df['Continent'] == continent]['ContinentId'].iloc[0]
    region_id       = df[df['Region'] == region]['RegionId'].iloc[0]
    country_id      = df[df['Country'] == country]['CountryId'].iloc[0]
    att_avg_rating  = df[df['AttractionId'] == attraction_id]['Attraction_Avg_Rating'].iloc[0]
    att_visit_count = df[df['AttractionId'] == attraction_id]['Attraction_Visit_Count'].iloc[0]

    if st.button("🧳 Predict Visit Mode", use_container_width=True):
        input_data = np.array([[
            visit_year, visit_month, attraction_id,
            att_type_id, continent_id, region_id,
            country_id, rating, user_avg_rating,
            user_visit_count, att_avg_rating, att_visit_count
        ]])
        prediction = clf_model.predict(input_data)[0]
        mode       = le.inverse_transform([prediction])[0]

        mode_emoji = {
            "Couples": "💑", "Family": "👨‍👩‍👧‍👦",
            "Friends": "👫", "Solo": "🧍", "Business": "💼"
        }
        emoji = mode_emoji.get(mode, "🧳")
        st.markdown("---")
        st.success(f"### Predicted Visit Mode: {emoji} {mode}")

# ── RECOMMENDATIONS ────────────────────────────────────────────────────────────
elif page == "🎯 Get Recommendations":
    st.title("🎯 Get Attraction Recommendations")
    st.markdown("---")

    user_id  = int(st.number_input("Enter User ID", min_value=1, value=70456))
    n_recs   = st.slider("Number of Recommendations", 3, 10, 5)
    rec_type = st.radio("Recommendation Type",
                        ["Collaborative Filtering",
                         "Content Based Filtering",
                         "Both (Hybrid)"],
                        horizontal=True)

    def collaborative_filtering(user_id, n_recommendations=5):
        if user_id not in user_matrix.index:
            return None
        # Find top 50 similar users using cosine similarity
        similar_users = pd.Series(
            cosine_similarity([user_matrix.loc[user_id]], user_matrix)[0],
            index=user_matrix.index
        ).sort_values(ascending=False)[1:51].index.tolist()

        # Attractions already visited by this user
        visited = set(df[df['UserId'] == user_id]['AttractionId'].values)

        # Collect ratings from similar users for unvisited attractions
        recommendations = {}
        for sim_user in similar_users:
            sim_att = df[df['UserId'] == sim_user]\
                .groupby('AttractionId')['Rating'].mean()
            for att_id, rating in sim_att.items():
                if att_id not in visited:
                    recommendations.setdefault(att_id, []).append(rating)

        # Fallback — recommend top rated attractions overall
        if not recommendations:
            top = df.groupby('AttractionId')['Rating']\
                .mean().sort_values(ascending=False)
            top = top[~top.index.isin(visited)].head(n_recommendations)
            att_names = df[['AttractionId', 'Attraction', 'AttractionType']]\
                .drop_duplicates('AttractionId')
            return pd.DataFrame({
                'AttractionId': top.index, 'Rating': top.values
            }).merge(att_names, on='AttractionId')[['Attraction', 'AttractionType', 'Rating']]

        # Rank by average rating from similar users
        rec_df = pd.DataFrame([
            {'AttractionId': k, 'Rating': np.mean(v)}
            for k, v in recommendations.items()
        ]).sort_values('Rating', ascending=False).head(n_recommendations)
        att_names = df[['AttractionId', 'Attraction', 'AttractionType']]\
            .drop_duplicates('AttractionId')
        return rec_df.merge(att_names, on='AttractionId')[
            ['Attraction', 'AttractionType', 'Rating']]

    def content_based_filtering(user_id, n_recommendations=5):
        # Get attractions visited by user with their ratings
        user_visited = df[df['UserId'] == user_id]\
            .groupby('AttractionId')['Rating'].mean()
        if user_visited.empty:
            return None

        # Find unvisited attractions
        visited_ids = user_visited.index.tolist()
        not_visited = [a for a in att_sim.index if a not in visited_ids]
        if not not_visited:
            return None

        # Score each unvisited attraction by weighted similarity
        scores = {}
        for attraction in not_visited:
            sim_scores = []
            for vid in visited_ids:
                if vid in att_sim.index:
                    # similarity score × user rating = weighted score
                    sim_scores.append(att_sim.loc[vid, attraction] * user_visited[vid])
            scores[attraction] = np.mean(sim_scores) if sim_scores else 0

        rec_df = pd.DataFrame([
            {'AttractionId': k, 'Score': v}
            for k, v in scores.items()
        ]).sort_values('Score', ascending=False).head(n_recommendations)
        att_names = df[['AttractionId', 'Attraction',
                        'AttractionType', 'Attraction_Avg_Rating']]\
            .drop_duplicates('AttractionId')
        return rec_df.merge(att_names, on='AttractionId')[
            ['Attraction', 'AttractionType', 'Attraction_Avg_Rating', 'Score']]

    if st.button("🎯 Get Recommendations", use_container_width=True):
        st.markdown("---")

        if rec_type in ["Collaborative Filtering", "Both (Hybrid)"]:
            st.subheader("📌 Collaborative Filtering")
            result = collaborative_filtering(user_id, n_recs)
            if result is not None:
                st.dataframe(result, use_container_width=True)
            else:
                st.warning("User not found!")

        if rec_type in ["Content Based Filtering", "Both (Hybrid)"]:
            st.subheader("📌 Content Based Filtering")
            result = content_based_filtering(user_id, n_recs)
            if result is not None:
                st.dataframe(result, use_container_width=True)
            else:
                st.warning("User not found or visited all attractions!")
