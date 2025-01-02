import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model  # type: ignore
from streamlit_lottie import st_lottie
import json


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the animation
lottie_animation = load_lottiefile("Animation.json")

# Initialize session 
if "page" not in st.session_state:
    st.session_state.page = "home"

# Database connection 
def get_db_connection():
    #engine = create_engine("mssql+pyodbc://sa:123@Sudhakar\\SQLEXPRESS01/Local_database?driver=ODBC+Driver+17+for+SQL+Server")
    engine = create_engine("mysql+mysqlconnector://admin:Sachin123#@localdb.cxkka06iqzik.ap-south-1.rds.amazonaws.com/Local_database")
    #engine = create_engine("mysql+mysqlconnector://root:sachin%400210@127.0.0.1:3306/gravity_books")
    return engine



if "model_loaded" not in st.session_state:
    st.session_state.model = load_model("book_genre_model.keras")
    with open("vectorizer.pkl", "rb") as f:
        st.session_state.vectorizer = pickle.load(f)
    with open("label_binarizer.pkl", "rb") as f:
        st.session_state.label_binarizer = pickle.load(f)
    st.session_state.model_loaded = True
    
# Page Navigation
if st.session_state.page == "home":
    st.title("Welcome to Book Recommendation System")  
       
    # Create layout with columns
    col1, col2 = st.columns([4, 1])  

    with col2:
        st.write("")
        st.write("")
        if st.button("➡ Search Book", key="go_button"):
            st.session_state.page = "details"
            
    with col1:
        # Display Animation
        st_lottie(lottie_animation, height=425, key="home_animation")
    

elif st.session_state.page == "details":
    st.title("Book Details and Recommendations")

    # Dropdown and input fields
    col1, col2 = st.columns(2)
    with col1:
        input_option = st.selectbox(
            "Select the input type:",
            ("Title", "Author", "Publisher Name", "Published Year"),
            key="input_option"
        )
    with col2:
        input_value = st.text_input(
            f"Enter {input_option}:",
            key="input_value"
        )

    # Prepare input
    def prepare_input(option, value):
        title, author, publisher, year = "", "", "", "Unknown"
        if option == "Title":
            title = value
        elif option == "Author":
            author = value
        elif option == "Publisher Name":
            publisher = value
        elif option == "Published Year":
            year = value
        return f"{title} {author} {publisher} {year}"

    # Fetch book details
    def fetch_book_details(option, value):
        column_mapping = {
            "Title": "title",
            "Author": "author_name",
            "Publisher Name": "publisher_name",
            "Published Year": "published_year",
        }
        column_name = column_mapping.get(option)
        if not column_name:
            st.error("Invalid input option!")
            return pd.DataFrame()
        search_value = f"%{value}%"
        query = f"""
            SELECT 
                title AS book_name, 
                author_name AS Author,
                Publisher_name AS Publisher, 
                published_year,
                genres,
                cover_image_url                
            FROM 
                book_data
            WHERE
                {column_name} Like %s
            GROUP BY 
                title, author_name, Publisher_name, published_year, genres,cover_image_url;
        """
        engine = get_db_connection()
        with engine.connect() as connection:
            return pd.read_sql(query, connection, params=(search_value,))

    
    # Fetch related books with a fallback
    def fetch_related_books(genres):
        genre_list = "','".join(genres)
        
        # Primary query
        primary_query = f"""
            SELECT 
                title AS book_name, 
                author_name AS Author,
                Publisher_name AS Publisher, 
                published_year,
                cover_image_url                
            FROM 
                book_data
            WHERE 
                genres IN ('{genre_list}')
            GROUP BY 
                title, author_name, Publisher_name, published_year, cover_image_url
            HAVING 
                COUNT(*) > 4 
            LIMIT 10
        """
        
        # Fallback query
        fallback_query = f"""
            SELECT 
                title AS book_name, 
                author_name AS Author,
                Publisher_name AS Publisher, 
                published_year,
                cover_image_url                
            FROM 
                book_data
            WHERE 
                genres IN ('{genre_list}')
            GROUP BY 
                title, author_name, Publisher_name, published_year, cover_image_url
            ORDER BY 
                book_id ASC
            LIMIT 10
        """
        
        # Establish database connection
        engine = get_db_connection()
        with engine.connect() as connection:
            # Execute primary query
            related_data = pd.read_sql(primary_query, connection)
            
            # fallback query
            if related_data.empty:
                related_data = pd.read_sql(fallback_query, connection)        
        return related_data  
    
    

    # Search functionality
    if st.button("Search Book"):
        if not input_value.strip():
            st.warning(f"Please enter a valid {input_option}!")
        else:
            single_input = prepare_input(input_option, input_value)
            single_input_tfidf = st.session_state.vectorizer.transform([single_input]).toarray()
            try:
            # Predict the genre
                prediction = st.session_state.model.predict(single_input_tfidf)
                predicted_genre = st.session_state.label_binarizer.inverse_transform((prediction > 0.5).astype(int))
            except IndexError:
                st.error("Prediction failed: Try different input.")
                predicted_genre = []

            if predicted_genre:
                st.write(f"Predicted Genres: {', '.join(predicted_genre)}")
                book_details = fetch_book_details(input_option, input_value)
                if not book_details.empty:
                    st.subheader("Book Details:")
                    st.dataframe(book_details)
                else:
                    st.write("No book details found.")

                related_books = fetch_related_books(predicted_genre)
                if not related_books.empty:
                    st.subheader("Recommended Books:")
                    st.dataframe(related_books)
                else:
                    st.write("No related books found.")
            else:
                st.write("No genres predicted. Please refine your input.")

    # # Back to home page
    # if st.button("⬅ Back to Home"):
    #     st.session_state.page = "home"
