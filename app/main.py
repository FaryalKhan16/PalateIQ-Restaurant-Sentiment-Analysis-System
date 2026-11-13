import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.database import get_connection, initialize_database
from app.restaurants import get_restaurant, get_restaurants

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "restaurant_sentiment_pipeline.joblib"
)

MODEL_INFO_PATH = (
    BASE_DIR
    / "models"
    / "model_info.json"
)

METRICS_PATH = (
    BASE_DIR
    / "results"
    / "three_class"
    / "final_test_metrics.json"
)

CLASSIFICATION_REPORT_PATH = (
    BASE_DIR
    / "results"
    / "three_class"
    / "final_test_classification_report.json"
)


# ============================================================
# LOAD DEPLOYMENT MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}. "
        "Run save_deployment_model.py first."
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Restaurant Review Sentiment Analyzer",
    description=(
        "A machine-learning API that analyzes restaurant reviews "
        "and classifies them into Negative, Neutral, or Positive "
        "sentiment."
    ),
    version="1.4.0",
    contact={
        "name": "Restaurant Sentiment Analysis Project",
    },
)

# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST MODELS
# ============================================================

class ReviewRequest(BaseModel):
    """
    Request body for analyzing one restaurant review.
    """

    text: str = Field(
        min_length=1,
        max_length=10000,
        description="Restaurant review text to analyze.",
        examples=[
            "The food was delicious and the service was excellent."
        ],
    )


class BatchReviewRequest(BaseModel):
    """
    Request body for analyzing multiple restaurant reviews.
    """

    reviews: list[str] = Field(
        min_length=1,
        max_length=50,
        description="List containing 1 to 50 restaurant reviews.",
        examples=[
            [
                "The food was delicious.",
                "The service was okay.",
                "The food was terrible."
            ]
        ],
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_review_history(
    review: str,
    sentiment: str,
    class_scores: dict,
) -> None:
    """
    Save one prediction to the review history database.
    """

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO review_history (
                review,
                sentiment,
                negative_score,
                neutral_score,
                positive_score,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                review,
                sentiment,
                class_scores.get("Negative"),
                class_scores.get("Neutral"),
                class_scores.get("Positive"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        connection.commit()

    finally:
        connection.close()


def analyze_review(review: str) -> dict:
    """
    Analyze one restaurant review.

    Returns:
        Review text, predicted sentiment, and class scores.
    """

    review = review.strip()

    if not review:
        raise HTTPException(
            status_code=422,
            detail="Review text cannot be blank.",
        )

    prediction = str(model.predict([review])[0])

    response = {
        "review": review,
        "sentiment": prediction,
    }

    class_scores = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review])[0]

        class_scores = {
            str(label): round(float(score), 4)
            for label, score in zip(
                model.classes_,
                probabilities,
            )
        }

        response["class_scores"] = class_scores

    save_review_history(
        review=review,
        sentiment=prediction,
        class_scores=class_scores,
    )

    return response


def read_json_file(file_path: Path) -> dict:
    """
    Read a JSON file and return its contents.
    """

    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Required data file not found: {file_path.name}",
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not read valid JSON from "
                f"{file_path.name}: {error}"
            ),
        )


def save_batch_review_history(
    reviews: list[str],
    predictions,
    all_probabilities,
) -> None:
    """
    Save multiple predictions to the review history database.
    """

    connection = get_connection()

    try:
        for index, (review, prediction) in enumerate(
            zip(reviews, predictions)
        ):

            class_scores = {}

            if all_probabilities is not None:
                class_scores = {
                    str(label): round(float(score), 4)
                    for label, score in zip(
                        model.classes_,
                        all_probabilities[index],
                    )
                }

            connection.execute(
                """
                INSERT INTO review_history (
                    review,
                    sentiment,
                    negative_score,
                    neutral_score,
                    positive_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    review,
                    str(prediction),
                    class_scores.get("Negative"),
                    class_scores.get("Neutral"),
                    class_scores.get("Positive"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        connection.commit()

    finally:
        connection.close()


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get(
    "/",
    summary="API Home",
    description="Check that the Restaurant Sentiment Analyzer API is running.",
)
def home():
    return {
        "message": "Restaurant Sentiment Analyzer API is running",
        "status": "success",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    summary="Health Check",
    description="Check whether the API and deployment model are available.",
)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get(
    "/model-info",
    summary="Get Model Information",
    description=(
        "Return information about the deployed sentiment "
        "classification model, its features, training data, "
        "and supported sentiment classes."
    ),
)
def get_model_info():
    return read_json_file(MODEL_INFO_PATH)


# ============================================================
# MODEL METRICS
# ============================================================

@app.get(
    "/metrics",
    summary="Get Model Evaluation Metrics",
    description=(
        "Return the saved final test metrics and detailed "
        "classification report for the deployed model."
    ),
)
def get_model_metrics():
    return {
        "final_test_metrics": read_json_file(METRICS_PATH),
        "classification_report": read_json_file(
            CLASSIFICATION_REPORT_PATH
        ),
    }


# ============================================================
# SINGLE REVIEW PREDICTION
# ============================================================

@app.post(
    "/predict",
    summary="Predict Sentiment for One Review",
    description=(
        "Analyze one restaurant review, save the result to "
        "history, and return its predicted sentiment together "
        "with class probability scores."
    ),
)
def predict_sentiment(request: ReviewRequest):
    return analyze_review(request.text)


# ============================================================
# BATCH REVIEW PREDICTION
# ============================================================

@app.post(
    "/predict/batch",
    summary="Predict Sentiment for Multiple Reviews",
    description=(
        "Analyze between 1 and 50 restaurant reviews in a single "
        "request. Each review receives a predicted sentiment and "
        "class probability scores, and all successful results "
        "are saved to history."
    ),
)
def predict_batch(request: BatchReviewRequest):

    # Clean the reviews
    cleaned_reviews = [
        review.strip()
        for review in request.reviews
    ]

    # Check for blank reviews
    if any(not review for review in cleaned_reviews):
        raise HTTPException(
            status_code=422,
            detail="Every review must contain non-blank text.",
        )

    # Check review length
    if any(
        len(review) > 10000
        for review in cleaned_reviews
    ):
        raise HTTPException(
            status_code=422,
            detail="Each review must contain at most 10000 characters.",
        )

    # Generate predictions
    predictions = model.predict(cleaned_reviews)

    results = []

    # Generate probabilities if supported
    if hasattr(model, "predict_proba"):
        all_probabilities = model.predict_proba(
            cleaned_reviews
        )
    else:
        all_probabilities = None

    # Build results
    for index, (review, prediction) in enumerate(
        zip(cleaned_reviews, predictions)
    ):

        result = {
            "review": review,
            "sentiment": str(prediction),
        }

        if all_probabilities is not None:
            result["class_scores"] = {
                str(label): round(float(score), 4)
                for label, score in zip(
                    model.classes_,
                    all_probabilities[index],
                )
            }

        results.append(result)

    # Save all successful predictions to history
    save_batch_review_history(
        reviews=cleaned_reviews,
        predictions=predictions,
        all_probabilities=all_probabilities,
    )

    return {
        "count": len(results),
        "results": results,
    }

# ============================================================
# REVIEW HISTORY
# ============================================================

@app.get(
    "/history",
    summary="Get Review History",
    description=(
        "Return previously analyzed restaurant reviews "
        "saved in the SQLite history database."
    ),
)
def get_review_history(
    limit: int = 20,
    offset: int = 0,
):
    """
    Return saved review predictions with pagination.
    """

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 100.",
        )

    if offset < 0:
        raise HTTPException(
            status_code=422,
            detail="Offset cannot be negative.",
        )

    connection = get_connection()

    try:
        total_row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM review_history
            """
        ).fetchone()

        rows = connection.execute(
            """
            SELECT
                id,
                review,
                sentiment,
                negative_score,
                neutral_score,
                positive_score,
                created_at
            FROM review_history
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        history = [
            dict(row)
            for row in rows
        ]

        total = total_row["total"]

        return {
            "count": len(history),
            "total": total,
            "limit": limit,
            "offset": offset,
            "history": history,
        }

    finally:
        connection.close()

# ============================================================
# HISTORY STATISTICS
# ============================================================

@app.get(
    "/history/stats",
    summary="Get History Statistics",
    description=(
        "Return statistics about analyzed restaurant reviews "
        "stored in the SQLite history database."
    ),
)
def get_history_stats():
    """
    Return total review count and sentiment distribution.
    """

    connection = get_connection()

    try:
        total_row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM review_history
            """
        ).fetchone()

        sentiment_rows = connection.execute(
            """
            SELECT
                sentiment,
                COUNT(*) AS count
            FROM review_history
            GROUP BY sentiment
            """
        ).fetchall()

        total = total_row["total"]

        sentiment_counts = {
            "Negative": 0,
            "Neutral": 0,
            "Positive": 0,
        }

        for row in sentiment_rows:
            sentiment = str(row["sentiment"])

            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] = row["count"]

        return {
            "total_reviews": total,
            "sentiment_counts": sentiment_counts,
        }

    finally:
        connection.close()


# ============================================================
# SINGLE HISTORY RECORD
# ============================================================

@app.get(
    "/history/{review_id}",
    summary="Get One History Record",
    description=(
        "Return one previously analyzed restaurant review "
        "from the SQLite history database."
    ),
)
def get_history_record(review_id: int):
    """
    Return one saved review prediction by its database ID.
    """

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                id,
                review,
                sentiment,
                negative_score,
                neutral_score,
                positive_score,
                created_at
            FROM review_history
            WHERE id = ?
            """,
            (review_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"History record with id {review_id} was not found.",
            )

        return dict(row)

    finally:
        connection.close()

 # ============================================================
# DELETE SINGLE HISTORY RECORD
# ============================================================

@app.delete(
    "/history/{review_id}",
    summary="Delete One History Record",
    description=(
        "Delete one previously analyzed restaurant review "
        "from the SQLite history database."
    ),
)
def delete_history_record(review_id: int):
    """
    Delete one saved review prediction by its database ID.
    """

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            DELETE FROM review_history
            WHERE id = ?
            """,
            (review_id,),
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=f"History record with id {review_id} was not found.",
            )

        connection.commit()

        return {
            "message": "History record deleted successfully.",
            "deleted_id": review_id,
        }

    finally:
        connection.close()  


# ============================================================
# CLEAR ALL HISTORY
# ============================================================

@app.delete(
    "/history",
    summary="Clear All History",
    description=(
        "Delete all previously analyzed restaurant reviews "
        "from the SQLite history database."
    ),
)
def clear_history():
    """
    Delete all saved review predictions.
    """

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            DELETE FROM review_history
            """
        )

        deleted_count = cursor.rowcount

        connection.commit()

        return {
            "message": "All history records deleted successfully.",
            "deleted_count": deleted_count,
        }

    finally:
        connection.close()

# ============================================================
# RESTAURANT EXPLORER
# ============================================================

@app.get(
    "/restaurants",
    summary="Explore Restaurants",
    description=(
        "Return restaurants from the Yelp-derived dataset with "
        "optional name search, city filtering, and pagination."
    ),
)
def list_restaurants(
    search: str | None = None,
    city: str | None = None,
    limit: int = 24,
    offset: int = 0,
):
    """
    Return a paginated list of restaurants.
    """

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 100.",
        )

    if offset < 0:
        raise HTTPException(
            status_code=422,
            detail="Offset cannot be negative.",
        )

    return get_restaurants(
        search=search,
        city=city,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/restaurants/{restaurant_name}",
    summary="Get Restaurant Details",
    description=(
        "Return restaurant information and customer reviews "
        "for a selected restaurant."
    ),
)
def restaurant_details(restaurant_name: str):
    """
    Return details and reviews for one restaurant.
    """

    restaurant = get_restaurant(restaurant_name)

    if restaurant is None:
        raise HTTPException(
            status_code=404,
            detail=f"Restaurant '{restaurant_name}' was not found.",
        )

    return restaurant
