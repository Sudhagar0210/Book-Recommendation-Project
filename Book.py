import streamlit as st
import pandas as pd
import pyodbc
import pickle
import numpy as np

# Database connection
def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=Sudhakar\\SQLEXPRESS01;'
        'DATABASE=Local_database;'
        'UID=sa;'
        'PWD=123'
    )
    return conn

@st.cache_resource
def load_model():
    with open("book_recommendation_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Predict the genre
def predict_genre(model, user_input):
    input_data = np.array(user_input).reshape(1, -1)
    predicted_genre = model.predict(input_data)
    return predicted_genre[0] 

# Fetch related books
def fetch_related_books(conn, genre):
    query = f"""
    SELECT TOP 10 *
    FROM Book_data
    WHERE genres LIKE '%{genre}%'
    ORDER BY order_id DESC
    """
    return pd.read_sql_query(query, conn)

# Fetch books based on entered fields
def fetch_books_by_fields(conn, book_name=None, author=None, publisher=None, book_year=None):
    query = "SELECT TOP 10 * FROM Book_data WHERE 1=1"
    if book_name:
        query += f" AND title LIKE '%{book_name}%'"
    if author:
        query += f" AND author_name LIKE '%{author}%'"
    if publisher:
        query += f" AND publisher_name LIKE '%{publisher}%'"
    if book_year:
        query += f" AND published_year = {book_year}"
    query += " ORDER BY order_id DESC"
    return pd.read_sql_query(query, conn)

# Streamlit app
def main():
    st.title("Book Recommendation System")
    st.write("Enter any details to predict the genre and get recommendations!")

    # Input box for all fields
    user_input = st.text_area(
        "Enter details (format: Book Name, Author, Publisher, Year):",
        placeholder="Example: Harry Potter, J.K. Rowling, Bloomsbury, 1997"
    )

    if st.button("Get Recommendation"):
        try:
            if user_input.strip():
                # Parse user input
                inputs = user_input.split(",")
                inputs = [i.strip() if i.strip() else None for i in inputs]

                # Extract fields and handle missing values
                book_name = inputs[0] if len(inputs) > 0 else None
                author = inputs[1] if len(inputs) > 1 else None
                publisher = inputs[2] if len(inputs) > 2 else None
                book_year = inputs[3] if len(inputs) > 3 and inputs[3].isdigit() else None

                # Display entered inputs
                st.write("**Entered Inputs:**")
                st.write(f"**Book Name:** {book_name or 'N/A'}")
                st.write(f"**Author:** {author or 'N/A'}")
                st.write(f"**Publisher:** {publisher or 'N/A'}")
                st.write(f"**Published Year:** {book_year or 'N/A'}")

                # Load the model
                model = load_model()
                if not hasattr(model, "predict"):
                    st.error("The loaded model is not a valid model object.")
                    return

                # Prepare user input for prediction (default values for missing fields)
                user_input_for_model = [
                    book_name or "Unknown",
                    author or "Unknown",
                    publisher or "Unknown",
                    int(book_year) if book_year else 2000,
                ]

                # Predict the genre
                try:
                    predicted_genre = predict_genre(model, user_input_for_model)
                    st.success(f"Predicted Genre: {predicted_genre}")
                except Exception as e:
                    st.warning("Could not predict genre. Fetching related books directly.")
                    predicted_genre = None

                # Fetch related books
                conn = get_db_connection()
                if predicted_genre:
                    related_books = fetch_related_books(conn, predicted_genre)
                else:
                    related_books = fetch_books_by_fields(conn, book_name, author, publisher, book_year)

                conn.close()

                if not related_books.empty:
                    st.write("Top Related Books:")
                    st.dataframe(related_books)
                else:
                    st.warning("No related books found!")
            else:
                st.warning("Please enter at least one detail to proceed!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()











# CREATE TABLE Book_Data (
#     id INT IDENTITY(1,1) PRIMARY KEY, 
#     book_id NVARCHAR(MAX),    
#     title NVARCHAR(MAX),           
#     isbn13 NVARCHAR(MAX),            
#     language_name NVARCHAR(MAX),               
#     publisher_name NVARCHAR(MAX),
# 	customer_id NVARCHAR(MAX),
# 	order_id NVARCHAR(MAX),
# 	author_name NVARCHAR(MAX),
# 	published_year NVARCHAR(MAX),
# 	genres NVARCHAR(MAX)
# )

# select * from Book_Data order by 1 asc




































# import streamlit as st
# import numpy as np
# import pyodbc
# import pickle

# # Load the trained model
# #model = load_model("book_recommendation_model.h5")

# model= "book_recommendation_model.pkl"

# with open(model, "wb") as file:
#     pickle.dump(model, file)


# def get_db_connection():
#     conn = pyodbc.connect(
#         'DRIVER={SQL Server};'
#         'SERVER=Sudhakar\\SQLEXPRESS01;'
#         'DATABASE=Local_database;'
#         'UID=sa;'
#         'PWD=123'
#     )
#     return conn

# # Streamlit app
# st.title("Book Recommendation System")

# # Input box for all fields
# user_input = st.text_area(
#     "Enter details (format: Book Name, Author, Publisher, Year):",
#     placeholder="Example: Harry Potter, J.K. Rowling, Bloomsbury, 1997"
# )

# # Button to get the recommendation
# if st.button("Get Recommendation"):
#     try:
#         if user_input.strip():
#             # Parse user input
#             inputs = user_input.split(",")
#             inputs = [i.strip() for i in inputs]

#             # Extract fields 
#             book_name = inputs[0] if len(inputs) > 0 else None
#             author = inputs[1] if len(inputs) > 1 else None
#             publisher = inputs[2] if len(inputs) > 2 else None
#             book_year = int(inputs[3]) if len(inputs) > 3 else 2000  

#             # Query the database 
#             conn = get_db_connection()
#             cursor = conn.cursor()

#             # Dynamic encoding using database data
#             def get_encoded_value(table, column, value):
#                 cursor.execute(f"SELECT rowid FROM {table} WHERE {column} = ?", (value,))
#                 result = cursor.fetchone()
#                 return result[0] if result else 0  # Default to 0 if not found

#             encoded_book_name = get_encoded_value("book", "title", book_name)
#             encoded_author = get_encoded_value("books", "author", author)
#             encoded_publisher = get_encoded_value("books", "publisher", publisher)

#             # Normalize book_year
#             normalized_year = (book_year - 1800) / (2024 - 1800)

#             # Combine inputs into a NumPy array
#             input_data = np.array([[encoded_book_name, encoded_author, encoded_publisher, normalized_year]]).astype('float32')

#             # Predict
#             prediction = model.predict(input_data)
#             recommended_genre = np.argmax(prediction)  # Replace with genre decoding logic

#             # Query database using the predicted genre
#             query = f"""
#             SELECT * FROM book_genres 
#             WHERE genre = ? 
#             LIMIT 5;
#             """
#             cursor.execute(query, (recommended_genre,))
#             results = cursor.fetchall()

#             # Display results
#             if results:
#                 st.write("Recommended Books:")
#                 for row in results:
#                     st.write(f"Title: {row[1]}, Author: {row[2]}, Publisher: {row[3]}, Year: {row[4]}")
#             else:
#                 st.write("No books found for the recommended genre.")

#             conn.close()
#         else:
#             st.warning("Please enter details in the text box.")

#     except Exception as e:
#         st.error(f"An error occurred: {e}")
