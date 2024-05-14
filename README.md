Movie Recommender System
This project implements a Movie Recommender System using Python. It leverages data preprocessing, natural language processing (NLP), and machine learning techniques to recommend movies based on user preferences. The system suggests 5 top-rated movies for a given input movie using cosine similarity.

Overview
The project is divided into several key steps:

Data Collection: The movie dataset is obtained from two CSV files: tmdb_5000_movies.csv and tmdb_5000_credits.csv.

Data Preprocessing: The raw data is cleaned and preprocessed to extract relevant features such as movie genres, keywords, cast, and crew.

Text Processing: The text data (genres, keywords, overview, cast, and crew) is processed using NLP techniques, including tokenization, stemming, and lowercasing.

Feature Extraction: CountVectorizer is used to convert the processed text data into numerical vectors.

Similarity Calculation: Cosine similarity is computed between the feature vectors to measure the similarity between movies.

Recommendation Generation: Based on the input movie, the system identifies similar movies and recommends the top 5 most similar ones.

Model Serialization: The final model, along with necessary data structures, is serialized using pickle for future use.


The system will output the top 5 recommended movies based on cosine similarity.

Files
recommend.py: Python script to recommend movies based on user input.
tmdb_5000_movies.csv: Dataset containing information about movies.
tmdb_5000_credits.csv: Dataset containing credits information for movies.
movies.pkl: Serialized DataFrame containing movie data.
similarity.pkl: Serialized cosine similarity matrix.
README.md: Documentation file explaining the project and its usage.

Dependencies
pandas
numpy
scikit-learn
nltk
Credits

This project was created by Shubham Tomar. The movie dataset used in this project is sourced from The Movie Database (TMDb) Kaggle. Special thanks to the contributors of the datasets and the open-source community.

Feel free to explore the code, contribute to the project, or use it for your own applications!

