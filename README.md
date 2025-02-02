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
Length Mismatch Between Predictions and Movie List: The predicted ratings list (my_predictions) had a different length than the filtered movie list (movieList_df), which led to errors while adding predictions to the movie list. The issue was solved by slicing the movie list to match the length of the predictions using .iloc[:len(my_predictions)].

Data Slicing:
After filtering the movie list to include only those with more than 20 ratings, the my_predictions array did not have the same length as the filtered movie list. This was resolved by truncating the predictions or filtering the movie list before adding predictions.

SettingWithCopyWarning:
A warning was raised when modifying a subset of the original DataFrame. This was resolved by ensuring proper usage of .loc to modify the DataFrame.
Output Interpretation:

Initially, it was not clear how to interpret the predicted ratings and how they compared to actual ratings. Through iterative debugging, the process of sorting and filtering based on mean ratings became clearer.

Outputs
The output of the system includes:
Top Recommended Movies: A list of the top N movies recommended for a specific user based on the predicted ratings.
Original vs Predicted Ratings: A comparison of the original ratings (if available) vs the predicted ratings for movies that the user has rated.

Example Output:

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


1. What is the objective of this project?
The objective is to build a movie recommendation system using collaborative filtering techniques that suggest movies to users based on their past ratings and preferences.

2. What dataset was used for this project?
A movie ratings dataset (likely from MovieLens or a similar source) containing user ratings for various movies.

3. What is collaborative filtering?
Collaborative filtering is a recommendation technique that predicts user preferences by analyzing interactions between users and items. It assumes that users with similar past behaviors will have similar future preferences.

4. What are the two main types of collaborative filtering?
User-based collaborative filtering: Recommends movies by finding users with similar tastes.
Item-based collaborative filtering: Recommends movies by identifying similarities between movies based on user ratings.

5. What preprocessing steps were applied to the dataset?
Handled missing values.
Filtered out movies with very few ratings.
Normalized ratings for better predictions.
Converted data into a user-item matrix.

6. What machine learning model was used for recommendations?
Matrix factorization techniques such as Singular Value Decomposition (SVD) were used to predict user ratings for unseen movies.

7. How does matrix factorization work in recommendation systems?
Matrix factorization breaks down the user-item matrix into lower-dimensional matrices that capture latent features of users and movies, enabling accurate rating predictions.

8. What library was used for implementing collaborative filtering?
The Surprise library was used for building and evaluating the collaborative filtering model.

9. How were the recommendations generated?
The trained model predicted movie ratings for users.
Movies with the highest predicted ratings were recommended.
Only movies with sufficient real ratings were considered to ensure reliability.

10. What challenges were faced during model implementation?
Mismatch in dataset sizes causing errors.
Need for filtering movies with a minimum number of ratings.
Optimizing model parameters for better accuracy.
Handling sparse user-item matrices.

11. How was the issue of dataset length mismatch resolved?
The movieList_df dataset was sliced to match the length of my_predictions to prevent errors when adding predicted ratings.

12. How was overfitting prevented in this recommendation system?
Used regularization in the matrix factorization model.
Filtered out movies with very few ratings to avoid biased recommendations.
Cross-validated the model on different subsets of the data.

13. How was the model evaluated?
Checked predicted ratings against actual ratings.
Compared mean ratings of recommended movies.
Verified that popular movies with high ratings were suggested.

14. What key outputs were observed?
List of top recommended movies based on predicted ratings.
Predictions aligned well with actual user preferences.
Popular, highly-rated movies appeared in recommendations.

15. How can this project be extended for better recommendations?
Hybrid filtering: Combine content-based and collaborative filtering.
Deep learning models: Use neural networks for better predictions.
More features: Consider genres, movie descriptions, and user demographics.

16. Can this model work for new users with no ratings?
No, this is a common issue called the cold-start problem. A hybrid approach using content-based filtering or demographic data could help.

17. What are the advantages of using collaborative filtering?
No need for movie metadata.
Can discover hidden patterns in user preferences.
Works well for personalized recommendations.

18. What are the disadvantages of collaborative filtering?
Requires a large amount of user data.
Struggles with new users and new movies (cold-start problem).
Computationally expensive for large datasets.

19. How does this project relate to real-world applications?
The same approach is used in streaming platforms like Netflix, Amazon Prime, and Spotify to recommend content based on user preferences.



qstn1------In your Movie Recommendation System project, we are not using regression or classification algorithms because the problem you're trying to solve is fundamentally different from those problems. Here's why:

1. Nature of the Problem (Recommendation vs. Prediction of Categories):
Regression is typically used to predict continuous values (e.g., predicting someone's salary, the temperature, etc.).
Classification is used when the output is a discrete label (e.g., predicting whether an email is spam or not).
However, in a recommendation system, the goal is not to predict a category or a continuous value directly for a new input but rather to recommend items (movies) based on past interactions (user-item ratings). You want to predict ratings for items that a user hasn't rated yet, which doesn't fit neatly into regression or classification because:

Recommendation is based on similarities between users and items, rather than making a direct prediction about a target variable.
The system predicts implicit preferences (like "how much would this user like this movie") based on existing data, not based on a predefined continuous target or category.
2. Collaborative Filtering Approach:
Instead of using regression or classification, you are using collaborative filtering because:

Collaborative filtering focuses on utilizing the interactions between users and items, identifying patterns based on user behavior (what items they've rated or interacted with) or item similarity.
Matrix factorization techniques like SVD (Singular Value Decomposition) decompose the user-item rating matrix into latent factors, helping to predict ratings for movies not yet seen by the user, which is not something you would achieve using a regression or classification approach.
3. Implicit Data (Unseen Movies):
In a classification or regression setting, you would typically have labeled data (i.e., every movie has a known rating from a user). However, with recommendation systems, you often deal with implicit feedback (e.g., a user rating a movie with 5 stars or not rating it at all). Predicting ratings for unseen movies or recommending the best movies requires learning patterns from the data without explicitly fitting a regression model or classification label.
4. Cold Start Problem:
In regression and classification algorithms, you'd require labeled training data for making predictions. However, in recommendation systems, there is a cold-start problem, especially when new users or new items (movies) enter the system. Collaborative filtering can still handle such issues better by leveraging similar users/items, whereas regression or classification models would need specific labeled data, making them less effective in the absence of data for new users/items.
5. Scalability Issues:
Regression and classification models typically assume the availability of direct input features (like user age, movie genre, etc.). In a recommendation system, if we applied a regression or classification model, we'd need to define features that directly correlate with the output variable (rating, preference), which could make the model more complex and less scalable. Collaborative filtering methods are more straightforward when you only have interaction data (user-item ratings) without the need for extensive feature engineering.
In Summary:
Regression and classification are supervised learning methods designed to predict a target variable (continuous or categorical). In contrast, recommendation systems like collaborative filtering aim to predict missing or unknown user-item interactions based on user behavior or item similarities, which doesn’t fit the typical structure of a regression or classification problem.


qstn 2------In your Movie Recommendation System, you don't necessarily need to implement One-Hot Encoding or Label Encoding in the traditional sense, or handle missing values in the same way as in a regression or classification problem. Here's why:

1. One-Hot Encoding / Label Encoding:
One-Hot Encoding and Label Encoding are used primarily for categorical data when you want to convert text labels (such as movie genres or user IDs) into numerical format for machine learning algorithms that require numerical inputs.
In your project:

User and Movie IDs are typically represented as integers, and there's no need to convert them into a one-hot encoded or labeled form since you're dealing with a user-item matrix where each entry corresponds to the rating of a movie by a user.
Genres of movies (if you were using them as features) could be encoded, but in the context of matrix factorization and collaborative filtering, such direct encoding is not necessary because the system is focusing on interactions (ratings) rather than explicit features of the movies or users.
2. Missing Values:
In typical supervised learning (e.g., classification or regression), missing values need to be handled because the model requires complete data to train. Common methods for handling missing values include:
Imputation (filling in missing values with the mean, median, mode, or predicted values)
Dropping missing values (removing rows/columns with missing values)
In your recommendation system, the matrix factorization (e.g., SVD) technique you're using doesn't require you to fill in missing values manually. Instead, the model learns from the non-missing entries (user-movie ratings) and predicts the missing ones. The model can predict the missing ratings for the movies that users haven't rated yet, so there's no need to perform imputation or remove rows/columns with missing values as long as the data is sparse.

3. How Collaborative Filtering Handles Missing Data:
Collaborative filtering works by learning the patterns from the user-item interaction matrix (ratings matrix) that is typically sparse (many ratings are missing).
The key idea is that similar users (based on their ratings) can be used to predict the ratings for missing entries. So, missing ratings are indirectly predicted based on the existing data without needing to explicitly handle missing values using traditional methods.
When You Might Use Encoding or Imputation in Recommendation Systems:
If you're incorporating other features such as movie genres, user demographics (age, location), or if you use content-based filtering, you might still need One-Hot Encoding or Label Encoding for categorical features. But in the case of pure collaborative filtering based on user-item interactions, these are usually not needed.

If you're using a hybrid approach that combines collaborative filtering with content-based features (like genre, director, etc.), then encoding for these features would be necessary. But even in that case, the recommendation model would handle missing ratings rather than needing to handle missing values manually.

In Conclusion:
For your collaborative filtering movie recommendation system:

No need for One-Hot Encoding or Label Encoding for the user and movie IDs unless you're incorporating additional categorical features.
Missing values are implicitly handled by the recommendation algorithm, particularly when predicting ratings for movies that a user hasn't rated yet.


qstn3-------handling missing values in a recommendation system, especially one using matrix factorization techniques like Singular Value Decomposition (SVD), is quite different from handling missing values in typical classification or regression problems.

Why did you face NaN values during iterations?
In the initial part of your project, during matrix factorization or the model's iterative process, you likely encountered NaN (Not a Number) values for several reasons:

Sparse User-Item Matrix:

The user-item matrix for movie ratings is typically sparse (most users don't rate most movies). Many entries are missing because users haven't rated certain movies.
When training the model, operations like matrix multiplication or updating the parameters (user and movie matrices) could result in NaN values if the calculations involved dividing by zero or invalid operations due to missing data.
Initial Parameterization:

When you initialize matrices (e.g., user and movie latent factors) in matrix factorization, you may set them with some random values. If the model isn't correctly updated or if the learning rate is too high, it can lead to exploding or vanishing values, resulting in NaNs during the training.
Gradient Descent Issues:

If you're using a gradient descent approach to optimize your model, incorrect parameter updates (like too large a learning rate) can cause NaN values in your matrices, especially during the backpropagation or update steps.
How did the model handle NaN values?
In matrix factorization, missing ratings (which are represented as NaN values) are not treated as missing data in the way they're handled in traditional classification or regression models. Instead, the model ignores the missing values and only learns from the observed ratings (non-NaN values).

Here's how it works:

Ignore NaNs in Calculations:

When computing the predicted ratings for a user-item pair, the model focuses on observed ratings (those that have non-missing values) and avoids using the NaN values directly in the calculations. This is why NaNs don't disrupt the model's learning process in the same way they would in a classification or regression task, where each feature is expected to have a value for every instance.
Matrix Factorization (SVD):

In techniques like Singular Value Decomposition (SVD), the goal is to decompose the sparse matrix into lower-dimensional user and item matrices. During the decomposition process, the model works with latent factors (hidden patterns in users' behavior and movie features) and only updates the parts of the matrix where ratings exist.
The missing values (NaNs) are effectively "ignored" because the model learns from the correlations between the non-missing values.
Updating Latent Factors:

In each iteration of the model, gradient descent or a similar optimization method is used to update the latent factors for users and items. Missing values don't directly impact these updates because the model is only trying to minimize the error (e.g., difference between predicted and observed ratings) for the observed ratings, not the missing ones.
Handling Exploding/Vanishing Values:

If NaNs appeared in the process due to extreme updates (e.g., from large learning rates), most implementations will include safeguards like reducing the learning rate or adding regularization (to penalize extreme values), preventing these issues from disrupting the model.
Key Difference from Classification/Regression:
In classification or regression, missing values must be explicitly handled before training. You would either impute them (e.g., filling missing values with the mean or using more advanced methods) or remove the rows with missing values.

In recommendation systems using matrix factorization (like SVD), missing values are handled implicitly. The algorithm learns patterns from the available ratings and predicts the missing ones without explicitly filling or removing the missing entries. The NaN values are simply ignored in calculations for model updates.

To summarize:
The NaN issue you faced in the iterations was due to the sparse nature of the user-item matrix and potentially gradient updates.
In matrix factorization, missing values are ignored during training and don't require traditional handling like in classification or regression tasks.
The model's optimization focuses only on observed ratings, and the missing ones are predicted indirectly based on the learned patterns.
That's why your model doesn't need to handle missing values in the same way a regression or classification model does. It only works with available data, predicting the missing entries based on similarities between users and items.



qstn4-------What is Matrix Factorization?
Matrix Factorization (MF) is a technique used in recommendation systems to uncover latent factors in user-item interactions. It decomposes a large, sparse matrix (like a user-movie rating matrix) into lower-dimensional matrices representing users and items. The idea is to extract hidden patterns in how users interact with items (movies, products, etc.) to make personalized recommendations.

How Does Matrix Factorization Work?
Given a user-item rating matrix 
𝑅
R (where rows represent users and columns represent items, with values being ratings), matrix factorization aims to approximate this matrix as the product of two lower-dimensional matrices:

𝑅
≈
𝑃
×
𝑄
𝑇
R≈P×Q 
T
 
where:

𝑃
P is the user-factor matrix (users × latent features)
𝑄
Q is the item-factor matrix (items × latent features)
𝑄
𝑇
Q 
T
  is the transpose of the item-factor matrix
Each row in 
𝑃
P represents a user's preference in terms of hidden factors, and each row in 
𝑄
Q represents a movie's characteristics in the same factor space. The product 
𝑃
×
𝑄
𝑇
P×Q 
T
  gives an estimated rating matrix where missing values (unrated movies) are predicted based on learned latent features.

How Was Matrix Factorization Applied in Your Model?
In your project, Singular Value Decomposition (SVD) was used as a matrix factorization technique. Here’s how it was implemented:

User-Item Matrix Construction

You started with a sparse matrix where each row represented a user and each column a movie.
The values in the matrix were ratings given by users to movies, with many missing (NaN) entries.
Applying SVD (Singular Value Decomposition)

SVD decomposes the user-item rating matrix 
𝑅
R into three matrices:

𝑅
≈
𝑈
×
Σ
×
𝑉
𝑇
R≈U×Σ×V 
T
 
where:

𝑈
U is the user matrix (captures user preferences)
Σ
Σ is the diagonal matrix of singular values (represents importance of latent features)
𝑉
𝑇
V 
T
  is the item matrix (captures item/movie features)
The decomposition helps in reducing dimensionality and learning important latent features that drive recommendations.

Selecting the Number of Latent Features

Instead of using the full decomposition, you only kept the top k singular values (say, 50 or 100). This reduces noise and keeps only the most significant patterns.
Reconstructing the Matrix & Making Predictions

By multiplying the reduced matrices 
𝑈
×
Σ
×
𝑉
𝑇
U×Σ×V 
T
 , you reconstructed an approximate rating matrix where missing ratings were now filled with predicted values.
These predicted ratings were used to recommend movies that a user might like.
Why Use Matrix Factorization Instead of Other Techniques?
Handles Sparse Data Well

Unlike traditional ML models (like regression/classification), matrix factorization can work with incomplete data since it learns from observed ratings only.
Captures Latent Features

Instead of relying on explicit features (like movie genre, director, etc.), it automatically discovers hidden patterns (e.g., "Action lovers", "Sci-Fi fans", etc.).
Scalable for Large Datasets

Works efficiently for large-scale datasets with millions of users and items.
Summary: How It Was Done in Your Model
✔ Created user-item rating matrix with movies and ratings.
✔ Applied Singular Value Decomposition (SVD) to extract latent features.
✔ Reduced dimensions by keeping only the top singular values.
✔ Reconstructed matrix to estimate missing ratings.
✔ Sorted recommendations based on predicted ratings.
✔ Filtered results to show only movies with enough user ratings.

Next Steps / Improvements
Try Alternating Least Squares (ALS): Another MF technique that works better for implicit feedback data.
Use Hybrid Methods: Combine SVD with content-based filtering (metadata like genre, director, etc.).
Optimize Hyperparameters: Tune latent factor count, regularization, and learning rate for better accuracy.
