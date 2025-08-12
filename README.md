import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

books_data = pd.read_csv('book.csv')

print("CSV Columns:", books_data.columns.tolist())

selected_features = ['authors', 'title', 'average_rating']

for feature in selected_features:
    if feature in books_data.columns:
        books_data[feature] = books_data[feature].fillna('')
    else:
        print(f"Warning: Missing column '{feature}'")

books_data = books_data[books_data['average_rating'] > 3.5]

combined_features = books_data['authors'] + ' ' + books_data['title']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

user_input = input("Enter a Book Title or Author: ")

user_input_vector = vectorizer.transform([user_input])

input_similarity = cosine_similarity(user_input_vector, feature_vectors).flatten()

best_match_index = np.argmax(input_similarity)
best_match = books_data.iloc[best_match_index]['title']

print(f"\nClosest Match Found: {best_match}")
sorted_books = sorted(
    list(enumerate(input_similarity)),
    key=lambda x: x[1], 
    reverse=True
)

print("\nTop 10 Recommended Books:\n")
for i, (book_index, score) in enumerate(sorted_books[1:11], start=1):
    title = books_data.iloc[book_index]['title']
    author = books_data.iloc[book_index]['authors']
    rating = books_data.iloc[book_index]['average_rating']
    print(f"{i}. {title} by {author} (Rating: {rating})")
