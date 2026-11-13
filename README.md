# 🍽️ PalateIQ – Restaurant Sentiment Analysis System

An AI-powered, full-stack restaurant review sentiment analysis platform that uses Natural Language Processing (NLP) and Machine Learning to understand customer opinions and classify restaurant reviews into **Positive, Neutral, and Negative** sentiments.

PalateIQ provides an interactive dashboard where users can analyze individual reviews, process multiple reviews, explore restaurant information, monitor sentiment statistics, and view model performance.

## 📌 Project Overview

Customer reviews play an important role in understanding restaurant performance and customer satisfaction. However, manually analyzing large amounts of feedback can be time-consuming.

PalateIQ addresses this problem by automatically analyzing restaurant reviews using machine-learning techniques and presenting the results through an interactive web interface.

The system predicts three sentiment categories:

- 😊 **Positive:** Indicates customer satisfaction or a favorable experience.
- 😐 **Neutral:** Indicates mixed, moderate, or neither strongly positive nor negative feedback.
- 😞 **Negative:** Indicates customer dissatisfaction or an unfavorable experience.

This project is a modified and extended version of an existing restaurant sentiment analysis project, with a different dataset, an improved three-class classification approach, and an expanded full-stack application.

## ✨ Key Features

- **Interactive Dashboard:** Displays overall review statistics and sentiment distribution.
- **Three-Class Sentiment Analysis:** Classifies reviews as Positive, Neutral, or Negative.
- **Individual Review Analysis:** Allows users to enter a restaurant review and receive an instant prediction.
- **Batch Review Analysis:** Analyzes multiple reviews in a single request.
- **Restaurant Explorer:** Allows users to explore restaurants, search by name, and filter by city.
- **Review History:** Stores previous predictions and allows users to search, view, and delete records.
- **Model Insights:** Displays model information and evaluation metrics.
- **Sentiment Probability Scores:** Provides confidence-related class probability scores for predictions.
- **Activity Timeline:** Visualizes review analysis activity over different time periods.
- **SQLite Database:** Stores analyzed reviews and their prediction results.

## 🧠 Machine Learning Approach

The project uses **Logistic Regression** as its deployed sentiment classification model.

### Model Details

| Component | Technique |
|---|---|
| Machine Learning Model | Logistic Regression |
| NLP Approach | Natural Language Processing |
| Feature Extraction | TF-IDF |
| Text Features | Word Unigrams and Bigrams |
| Text Preprocessing | Lowercasing, stopword removal, lemmatization |
| Classification | Three-Class Classification |
| Output Classes | Negative, Neutral, Positive |
| Model Serialization | Joblib |

### Text Preprocessing

The review text goes through preprocessing before being converted into numerical features.

The preprocessing steps include:

1. Converting text into lowercase.
2. Expanding common negative contractions.
3. Removing punctuation and unnecessary characters.
4. Removing English stopwords while preserving important negation words.
5. Applying lemmatization to normalize words.
6. Converting text into numerical features using TF-IDF.

### Model Training and Selection

The dataset was divided into three subsets:

- Training Set: 70%
- Validation Set: 15%
- Testing Set: 15%

Multiple machine-learning approaches were evaluated during model development. Logistic Regression was selected using validation Macro F1-score.

The final test set was kept separate from model selection and training.

## 📊 Dataset

The project uses the dataset:

**Yelp Restaurant Sentiment Master Dataset**

Dataset file: `yelp_sentiment_master_dataset.csv`

The dataset contains restaurant reviews and associated information, including review text, sentiment labels, restaurant names, locations, ratings, and categories.

### Dataset Statistics

| Description | Count |
|---|---:|
| Original Records | 14,351 |
| Duplicate Reviews Removed | 1 |
| Final Clean Records | 14,350 |
| Training Records | 10,045 |
| Validation Records | 2,152 |
| Testing Records | 2,153 |

### Sentiment Categories

- Negative
- Neutral
- Positive

*Note: The dataset included in this project is different from the dataset used in the original reference project.*

## 📈 Model Performance

The deployed Logistic Regression model was evaluated on a separate final test set.

### Final Test Results

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 82.49% |
| Macro Precision | 70.72% |
| Macro Recall | 73.51% |
| Macro F1-Score | 71.96% |
| Weighted F1-Score | 83.07% |

### Class-Wise F1 Scores

| Sentiment | F1-Score |
|---|---:|
| Negative | 80.29% |
| Neutral | 44.79% |
| Positive | 90.80% |

The model performs particularly well on positive and negative reviews. Neutral reviews remain more challenging because they can contain mixed opinions and less distinctive sentiment-related language.

## 🖥️ Technology Stack

### Frontend
- React.js
- JavaScript
- HTML5
- CSS3
- Vite
- Recharts
- Lucide React

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning & NLP
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Joblib

### Database
- SQLite

### Development Tools
- Visual Studio Code
- Git
- GitHub

## 🏗️ Project Architecture

The application follows a frontend-backend architecture.

```text
                 USER
                   |
                   v
          REACT FRONTEND
             (Vite)
                   |
                   v
            FASTAPI BACKEND
                   |
          -------------------
          |                 |
          v                 v
    ML SENTIMENT         SQLITE
      MODEL             DATABASE
          |
          v
     TF-IDF FEATURES
          |
          v
    LOGISTIC REGRESSION
          |
          v
   SENTIMENT PREDICTION
          |
          v
  NEGATIVE / NEUTRAL / POSITIVE
          |
          v
    DASHBOARD RESULTS
```

## 📂 Project Structure

```text
RestaurantSentimentProject/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── restaurants.py
│   │   └── ml/
│   │       ├── preprocess.py
│   │       ├── trainer.py
│   │       └── vectorize.py
│   │
│   ├── data/
│   │   ├── yelp_sentiment_master_dataset.csv
│   │   └── processed/
│   │
│   ├── models/
│   │   ├── restaurant_sentiment_pipeline.joblib
│   │   └── model_info.json
│   │
│   ├── results/
│   │   └── three_class/
│   │
│   ├── requirements.txt
│   └── run.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── api.js
│   │   └── restaurantImages.js
│   │
│   ├── package.json
│   └── index.html
│
└── README.md
```

## ⚙️ Installation and Setup

Follow these instructions to run the project locally.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd RestaurantSentimentProject
```

### 2. Backend Setup

Open a terminal and navigate to the backend folder:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the backend server:

```bash
python run.py
```

The backend will run at:

`http://127.0.0.1:8000`

FastAPI documentation:

`http://127.0.0.1:8000/docs`

### 3. Frontend Setup

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the frontend development server:

```bash
npm run dev
```

Open the application in your browser using the local URL displayed by Vite, usually:

`http://localhost:5173`

**Important:** Keep both frontend and backend servers running while using the application.

## 🔌 API Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Backend health check |
| GET | `/model-info` | Model information |
| GET | `/metrics` | Model evaluation metrics |
| POST | `/predict` | Analyze one review |
| POST | `/predict/batch` | Analyze multiple reviews |
| GET | `/history` | Retrieve review history |
| GET | `/history/stats` | Retrieve sentiment statistics |
| GET | `/history/{review_id}` | Retrieve a specific review |
| DELETE | `/history/{review_id}` | Delete a review record |
| DELETE | `/history` | Clear review history |
| GET | `/restaurants` | Explore restaurants |
| GET | `/restaurants/{restaurant_name}` | Retrieve restaurant details |

## 🎯 Project Objectives

1. Automate restaurant review sentiment classification.
2. Apply NLP techniques to process customer feedback.
3. Classify reviews into three sentiment categories.
4. Provide an interactive dashboard for sentiment monitoring.
5. Store and manage analyzed customer reviews.
6. Present machine-learning evaluation results.
7. Help understand customer satisfaction and identify areas for improvement.

## 🚀 Future Improvements

- Improve neutral and mixed-sentiment classification.
- Enhance sarcasm and context detection.
- Integrate real-time restaurant review sources.
- Add aspect-based sentiment analysis for food, service, price, and atmosphere.
- Explore advanced deep-learning and transformer-based models.
- Deploy the application to a cloud platform.

## 📚 Reference Project

This project is a modified and extended implementation inspired by the following repository:

[Sentiment Analysis of Restaurant Reviews – Original Reference Project](https://github.com/FaryalKhan16/Sentiment-Analysis-of-Restaurant-Reviews.git)

The current implementation uses a different dataset and includes an expanded three-class sentiment analysis system, an interactive React dashboard, a FastAPI backend, restaurant exploration, batch predictions, and review history management.

## 👩‍💻 Author

**Faryal Khan**

Developed as an academic machine-learning and full-stack development project.

---
