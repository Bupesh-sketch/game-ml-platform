from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Steam Game ML Platform",
    description="Game recommendation and price prediction API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PersonalizedRequest(BaseModel):
    liked_games: list[str]
    n: int = 10


# ============================================================
# MODEL PATH
# ============================================================

MODELS_PATH = (
    r"C:\Users\bupes\OneDrive\Desktop"
    r"\model\game_ml_platform\models"
)


# ============================================================
# LOAD RECOMMENDATION MODELS
# ============================================================

print("Loading recommendation models...")

tfidf_vectorizer = joblib.load(
    os.path.join(
        MODELS_PATH,
        "tfidf_vectorizer.pkl"
    )
)

recommendation_model = joblib.load(
    os.path.join(
        MODELS_PATH,
        "recommendation_model.pkl"
    )
)

games_ml = joblib.load(
    os.path.join(
        MODELS_PATH,
        "games_ml.pkl"
    )
)

print("Recommendation models loaded successfully!")


# ============================================================
# LOAD PRICE MODEL
# ============================================================

print("Loading price prediction model...")

price_model = joblib.load(
    os.path.join(
        MODELS_PATH,
        "price_model.pkl"
    )
)

price_features = joblib.load(
    os.path.join(
        MODELS_PATH,
        "price_features.pkl"
    )
)

print("Price model loaded successfully!")

print("Expected price features:")
print(price_features)

print("All ML models loaded successfully!")


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Steam Game ML Platform API is running!"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "recommendation_model": "loaded",
        "price_model": "loaded"
    }


# ============================================================
# GAME RECOMMENDATION
# ============================================================

@app.get("/recommend")
def recommend(
    game_name: str,
    n: int = 10
):

    # Validate game name
    if not game_name.strip():

        raise HTTPException(
            status_code=400,
            detail="game_name cannot be empty"
        )

    # Validate number of recommendations
    if n < 1 or n > 20:

        raise HTTPException(
            status_code=400,
            detail="n must be between 1 and 20"
        )

    # Find game
    matches = games_ml[
        games_ml["Name"].str.contains(
            game_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    # Game not found
    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=f"No game found matching '{game_name}'"
        )

    # Get first matching game
    game_index = matches.index[0]

    # Get recommendation feature text
    game_features = games_ml.loc[
        game_index,
        "features"
    ]

    # Convert game into TF-IDF vector
    game_vector = tfidf_vectorizer.transform(
        [game_features]
    )

    # Find nearest games
    distances, indices = recommendation_model.kneighbors(
        game_vector,
        n_neighbors=n + 1
    )

    # Remove the input game itself
    recommended_indices = indices[0][1:]

    similarities = 1 - distances[0][1:]

    recommendations = []

    for index, similarity in zip(
        recommended_indices,
        similarities
    ):

        game = games_ml.iloc[index]

        recommendations.append({

            "AppID": int(
                game["AppID"]
            ),

            "Name": game["Name"],

            "Genres": game["Genres"],

            "Tags": game["Tags"],

            "Price": float(
                game["Price"]
            ),

            "similarity": round(
                float(similarity),
                4
            )
        })

    return {

        "input_game": games_ml.loc[
            game_index,
            "Name"
        ],

        "recommendations": recommendations
    }


# ============================================================
# ORIGINAL MANUAL PRICE PREDICTION
# ============================================================

@app.get("/predict-price")
def predict_price(

    metacritic_score: float = 0,

    user_score: float = 0,

    positive: int = 0,

    negative: int = 0,

    peak_ccu: int = 0,

    release_year: int = 2026,

    genre_count: int = 0,

    tag_count: int = 0,

    category_count: int = 0,

    developer_count: int = 0,

    publisher_count: int = 0

):

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if positive < 0 or negative < 0:

        raise HTTPException(
            status_code=400,
            detail="positive and negative reviews cannot be negative"
        )

    if peak_ccu < 0:

        raise HTTPException(
            status_code=400,
            detail="peak_ccu cannot be negative"
        )

    if release_year < 1970 or release_year > 2026:

        raise HTTPException(
            status_code=400,
            detail="release_year must be between 1970 and 2026"
        )

    # --------------------------------------------------------
    # Derived features
    # --------------------------------------------------------

    total_reviews = (
        positive +
        negative
    )

    if total_reviews > 0:

        positive_ratio = (
            positive /
            total_reviews
        )

    else:

        positive_ratio = 0

    game_age = max(
        2026 - release_year,
        0
    )

    # --------------------------------------------------------
    # Feature order MUST match training
    # --------------------------------------------------------

    input_data = [[

        metacritic_score,

        user_score,

        positive,

        negative,

        peak_ccu,

        release_year,

        total_reviews,

        positive_ratio,

        game_age,

        genre_count,

        tag_count,

        category_count,

        developer_count,

        publisher_count

    ]]

    # --------------------------------------------------------
    # Predict log(price)
    # --------------------------------------------------------

    prediction_log = price_model.predict(
        input_data
    )[0]

    # Convert log1p(price) back to price
    predicted_price = np.expm1(
        prediction_log
    )

    # Prevent negative price
    predicted_price = max(
        float(predicted_price),
        0
    )

    return {

        "predicted_price": round(
            predicted_price,
            2
        ),

        "currency": "USD"
    }


# ============================================================
# GAME-BASED PRICE PREDICTION
# ============================================================

@app.get("/predict-game-price")
def predict_game_price(
    game_name: str
):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not game_name.strip():

        raise HTTPException(
            status_code=400,
            detail="game_name cannot be empty"
        )

    # --------------------------------------------------------
    # Find game
    # --------------------------------------------------------

    matches = games_ml[
        games_ml["Name"].str.contains(
            game_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=f"No game found matching '{game_name}'"
        )

    # Use first matching game
    game = matches.iloc[0]

    # --------------------------------------------------------
    # Basic features
    # --------------------------------------------------------

    metacritic_score = float(
        game["Metacritic score"]
    )

    user_score = float(
        game["User score"]
    )

    positive = int(
        game["Positive"]
    )

    negative = int(
        game["Negative"]
    )

    peak_ccu = int(
        game["Peak CCU"]
    )

    # --------------------------------------------------------
    # Total reviews
    # --------------------------------------------------------

    total_reviews = (
        positive +
        negative
    )

    # --------------------------------------------------------
    # Positive review ratio
    # --------------------------------------------------------

    if total_reviews > 0:

        positive_ratio = (
            positive /
            total_reviews
        )

    else:

        positive_ratio = 0

    # --------------------------------------------------------
    # Release date
    #
    # IMPORTANT:
    # Use the same logic as the notebook.
    # --------------------------------------------------------

    release_date = pd.to_datetime(
        game["Release date"],
        format="mixed",
        errors="coerce"
    )

    if pd.isna(release_date):

        release_year = 2026

    else:

        release_year = int(
            release_date.year
        )

    # --------------------------------------------------------
    # Game age
    # --------------------------------------------------------

    game_age = max(
        2026 - release_year,
        0
    )

    # ========================================================
    # CATEGORICAL COUNT FEATURES
    #
    # IMPORTANT:
    #
    # The notebook first replaced commas with spaces.
    #
    # Therefore the trained v3/v4 model uses:
    #
    #     .str.split().str.len()
    #
    # NOT:
    #
    #     .str.split(",").str.len()
    #
    # ========================================================

    genres = str(
        game["Genres"]
    ).strip()

    tags = str(
        game["Tags"]
    ).strip()

    categories = str(
        game["Categories"]
    ).strip()

    developers = str(
        game["Developers"]
    ).strip()

    publishers = str(
        game["Publishers"]
    ).strip()

    # --------------------------------------------------------
    # Count features
    # --------------------------------------------------------

    genre_count = (
        len(genres.split())
        if genres
        else 0
    )

    tag_count = (
        len(tags.split())
        if tags
        else 0
    )

    category_count = (
        len(categories.split())
        if categories
        else 0
    )

    developer_count = (
        len(developers.split())
        if developers
        else 0
    )

    publisher_count = (
        len(publishers.split())
        if publishers
        else 0
    )

    # --------------------------------------------------------
    # Create feature DataFrame
    #
    # This makes the feature order explicit.
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[

            metacritic_score,

            user_score,

            positive,

            negative,

            peak_ccu,

            release_year,

            total_reviews,

            positive_ratio,

            game_age,

            genre_count,

            tag_count,

            category_count,

            developer_count,

            publisher_count

        ]],
        columns=price_features
    )

    # --------------------------------------------------------
    # Predict log(price)
    # --------------------------------------------------------

    prediction_log = price_model.predict(
        input_data
    )[0]

    # --------------------------------------------------------
    # Convert log prediction back to dollars
    #
    # Model was trained with:
    #
    #     np.log1p(price)
    #
    # Therefore inference uses:
    #
    #     np.expm1(prediction)
    # --------------------------------------------------------

    predicted_price = np.expm1(
        prediction_log
    )

    # Prevent negative values
    predicted_price = max(
        float(predicted_price),
        0
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "game": game["Name"],

        "actual_price": round(
            float(game["Price"]),
            2
        ),

        "predicted_price": round(
            predicted_price,
            2
        ),

        "currency": "USD"

    }


# ============================================================
# PERSONALIZED RECOMMENDATION
# ============================================================

@app.post("/personalized-recommend")
def personalized_recommend(
    request: PersonalizedRequest
):

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    if not request.liked_games:

        raise HTTPException(
            status_code=400,
            detail="liked_games cannot be empty"
        )

    if request.n < 1 or request.n > 20:

        raise HTTPException(
            status_code=400,
            detail="n must be between 1 and 20"
        )

    # --------------------------------------------------------
    # Find liked games
    # --------------------------------------------------------

    liked_indices = []

    for game_name in request.liked_games:

        if not game_name.strip():

            continue

        matches = games_ml[
            games_ml["Name"].str.contains(
                game_name,
                case=False,
                na=False,
                regex=False
            )
        ]

        if not matches.empty:

            liked_indices.append(
                matches.index[0]
            )

    # No games found
    if not liked_indices:

        raise HTTPException(
            status_code=404,
            detail="None of the liked games were found"
        )

    # --------------------------------------------------------
    # Convert liked games into TF-IDF vectors
    # --------------------------------------------------------

    liked_vectors = (
        tfidf_vectorizer.transform(
            games_ml.iloc[
                liked_indices
            ]["features"]
        )
    )

    # --------------------------------------------------------
    # Create user profile
    # --------------------------------------------------------

    user_profile = np.asarray(
        liked_vectors.mean(
            axis=0
        )
    )

    # --------------------------------------------------------
    # Candidate recommendations
    # --------------------------------------------------------

    candidate_count = min(
        len(games_ml),
        request.n +
        len(liked_indices) +
        20
    )

    distances, indices = (
        recommendation_model.kneighbors(
            user_profile,
            n_neighbors=candidate_count
        )
    )

    # --------------------------------------------------------
    # Existing liked AppIDs
    # --------------------------------------------------------

    liked_appids = (
        games_ml.iloc[
            liked_indices
        ]["AppID"].tolist()
    )

    # --------------------------------------------------------
    # Build recommendations
    # --------------------------------------------------------

    recommendations = []

    for index, distance in zip(
        indices[0],
        distances[0]
    ):

        game = games_ml.iloc[index]

        # Don't recommend already liked games
        if game["AppID"] in liked_appids:

            continue

        recommendations.append({

            "AppID": int(
                game["AppID"]
            ),

            "Name": game["Name"],

            "Genres": game["Genres"],

            "Tags": game["Tags"],

            "Price": float(
                game["Price"]
            ),

            "similarity": round(
                float(1 - distance),
                4
            )

        })

        if len(recommendations) >= request.n:

            break

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "liked_games": games_ml.iloc[
            liked_indices
        ]["Name"].tolist(),

        "recommendations": recommendations

    }