Movie Recommendation System using Collaborative Filtering

Project Overview
This project implements a movie recommendation system using collaborative filtering techniques. The system aims to predict the ratings for movies that a user has not yet rated based on their previous ratings and the ratings of other users. The final recommendations are filtered and sorted based on the mean ratings of movies with sufficient data.

Key Features
Collaborative Filtering: A technique that makes predictions for a user based on the preferences of other similar users.
Matrix Factorization: Factorizes the user-item rating matrix into two lower-dimensional matrices, representing users and items.
Movie Recommendations: Provides top movie recommendations for a given user based on predicted ratings.
Mean Rating Filtering: Filters movies that have more than 20 ratings and sorts them by their mean ratings.

Inputs
The following datasets are used in the project:

ratings.csv: Contains user ratings for movies (userId, movieId, rating, timestamp).
movies.csv: Contains movie information (movieId, title, genre).
tags.csv: Contains tags for movies (userId, movieId, tag, timestamp).
links.csv: Contains IMDb and TMDb links for movies (movieId, IMDbId, TMDbId).

Methodology
Data Loading:

All datasets (ratings.csv, movies.csv, tags.csv, links.csv) are loaded using pandas.
Merged the ratings and movies datasets to combine ratings with movie titles.
Matrix Factorization:

Applied collaborative filtering using matrix factorization to predict missing ratings.
The user-item rating matrix was factorized into two matrices, X (user features) and W (movie features), along with a bias term b.
Prediction:

Predictions were made for a user’s ratings on movies they haven’t rated yet using the formula p = X @ W.T + b.
The predicted ratings were restored by adding the mean rating (Ymean).
Movie Recommendations:

Top N movies were recommended based on the predicted ratings.
Sorted the predictions in descending order to identify the most highly recommended movies.
Mean Rating Filtering:

Filtered out movies with less than 20 ratings to focus on more popular movies.
Added predictions to the movie list and sorted movies by their mean ratings.
Problems Faced
Length Mismatch Between Predictions and Movie List:

The predicted ratings list (my_predictions) had a different length than the filtered movie list (movieList_df), which led to errors while adding predictions to the movie list.
The issue was solved by slicing the movie list to match the length of the predictions using .iloc[:len(my_predictions)].
Data Slicing:

After filtering the movie list to include only those with more than 20 ratings, the my_predictions array did not have the same length as the filtered movie list.
This was resolved by truncating the predictions or filtering the movie list before adding predictions.
SettingWithCopyWarning:

A warning was raised when modifying a subset of the original DataFrame. This was resolved by ensuring proper usage of .loc to modify the DataFrame.
Output Interpretation:

Initially, it was not clear how to interpret the predicted ratings and how they compared to actual ratings. Through iterative debugging, the process of sorting and filtering based on mean ratings became clearer.
Outputs
The output of the system includes:

Top Recommended Movies: A list of the top N movies recommended for a specific user based on the predicted ratings.
Original vs Predicted Ratings: A comparison of the original ratings (if available) vs the predicted ratings for movies that the user has rated.
Example Output:
yaml
Copy
Edit
Top recommended movies based on mean rating and more than 20 ratings:
          pred  mean_rating  number_of_ratings  title
4313  3.624386     4.300000                 25  In the Name of the Father (1993)
4018  3.606284     4.293103                 29  Hoop Dreams (1994)
3499  2.853266     4.289062                192  Godfather, The (1972)
3782  3.040849     4.288462                 26  Harold and Maude (1971)
3011  3.845543     4.272936                218  Fight Club (1999)
Things to Remember
Data Preprocessing: Data cleaning and preprocessing (such as handling missing values and merging datasets) are crucial steps for building a robust recommendation system.
Collaborative Filtering: This method works well when there are enough interactions (ratings) between users and items. If the dataset is sparse, matrix factorization might struggle to provide accurate predictions.
Evaluation: The system has not been evaluated with metrics like RMSE or MAE. It's important to measure the accuracy of the model to understand how well it performs.
Model Tuning: Further tuning of the collaborative filtering model and its hyperparameters (e.g., number of latent factors, learning rate) can improve the recommendation quality.
Future Work
User Personalization: Incorporating user-specific features such as demographics or past behavior could further improve the quality of the recommendations.
Hybrid Models: Combining collaborative filtering with content-based filtering (e.g., genre, director, cast) could yield better recommendations, especially for new or obscure movies.
Model Optimization: Further optimization of the matrix factorization process, possibly using advanced techniques such as regularization, could prevent overfitting and improve prediction accuracy.
Real-time Recommendations: Implementing a system where new ratings and interactions update the model in real-time.
Installation and Dependencies
To run this project, you will need the following Python libraries:

pandas
numpy
tensorflow
matplotlib (optional, for visualizations)
You can install these dependencies using:
pip install pandas numpy tensorflow matplotlib
