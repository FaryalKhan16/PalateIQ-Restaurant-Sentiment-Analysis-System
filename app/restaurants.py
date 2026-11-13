import pandas as pd
from pathlib import Path


# ============================================================
# RESTAURANT DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "yelp_sentiment_master_dataset.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise RuntimeError(
        f"Restaurant dataset not found: {DATASET_PATH}"
    )


df = pd.read_csv(DATASET_PATH)


# ============================================================
# CLEAN BASIC VALUES
# ============================================================

df["name"] = df["name"].fillna("").astype(str).str.strip()
df["text"] = df["text"].fillna("").astype(str).str.strip()
df["categories"] = df["categories"].fillna("").astype(str).str.strip()
df["city"] = df["city"].fillna("").astype(str).str.strip()
df["state"] = df["state"].fillna("").astype(str).str.strip()


# ============================================================
# RESTAURANT SUMMARY
# ============================================================

def get_restaurants(
    search: str | None = None,
    city: str | None = None,
    limit: int = 24,
    offset: int = 0,
):
    """
    Return grouped restaurant information for Restaurant Explorer.
    """

    filtered = df

    # --------------------------------------------------------
    # Search by restaurant name
    # --------------------------------------------------------

    if search:
        search_value = search.strip().lower()

        if search_value:
            filtered = filtered[
                filtered["name"]
                .str.lower()
                .str.contains(search_value, na=False)
            ]

    # --------------------------------------------------------
    # Filter by city
    # --------------------------------------------------------

    if city:
        city_value = city.strip().lower()

        if city_value:
            filtered = filtered[
                filtered["city"]
                .str.lower()
                .eq(city_value)
            ]

    # --------------------------------------------------------
    # Group reviews into restaurants
    # --------------------------------------------------------

    grouped = (
        filtered
        .groupby(
            ["name", "city", "state"],
            dropna=False
        )
        .agg(
            review_count=("text", "count"),
            average_review_rating=("stars_review", "mean"),
            business_rating=("stars_business", "mean"),
            categories=("categories", "first"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Clean category display
    # --------------------------------------------------------

    grouped["categories"] = (
        grouped["categories"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Round numeric values
    # --------------------------------------------------------

    grouped["average_review_rating"] = (
        grouped["average_review_rating"]
        .round(1)
    )

    grouped["business_rating"] = (
        grouped["business_rating"]
        .round(1)
    )

    # --------------------------------------------------------
    # Sort restaurants by number of reviews
    # --------------------------------------------------------

    grouped = grouped.sort_values(
        by=["review_count", "name"],
        ascending=[False, True]
    )

    total = len(grouped)

    # --------------------------------------------------------
    # Pagination
    # --------------------------------------------------------

    paginated = grouped.iloc[
        offset: offset + limit
    ]

    restaurants = []

    for _, row in paginated.iterrows():

        restaurants.append(
            {
                "name": row["name"],
                "city": row["city"],
                "state": row["state"],
                "categories": row["categories"],
                "review_count": int(row["review_count"]),
                "average_review_rating": (
                    float(row["average_review_rating"])
                ),
                "business_rating": (
                    float(row["business_rating"])
                ),
            }
        )

    return {
        "count": len(restaurants),
        "total": total,
        "limit": limit,
        "offset": offset,
        "restaurants": restaurants,
    }


# ============================================================
# RESTAURANT DETAILS
# ============================================================

def get_restaurant(
    restaurant_name: str,
):
    """
    Return restaurant information and its customer reviews.
    """

    name_value = restaurant_name.strip().lower()

    matches = df[
        df["name"]
        .str.lower()
        .eq(name_value)
    ]

    if matches.empty:
        return None

    first_row = matches.iloc[0]

    reviews = []

    for _, row in matches.iterrows():

        reviews.append(
            {
                "review": row["text"],
                "stars_review": int(row["stars_review"]),
                "date": row["date"],
                "sentiment": row["rating_review"],
            }
        )

    categories = str(first_row["categories"])

    return {
        "name": first_row["name"],
        "city": first_row["city"],
        "state": first_row["state"],
        "categories": categories,
        "business_rating": round(
            float(matches["stars_business"].mean()),
            1
        ),
        "review_count": len(reviews),
        "reviews": reviews,
    }