from datetime import datetime
import streamlit as st
import requests

# API endpoints
POST_API_URL = "http://127.0.0.1:8000/predict/"
GET_API_URL = "http://127.0.0.1:8000/get_filtered_predict/"

# List of random reviews
random_reviews = [
    "I couldn't believe I wasted hours on this book. It was excruciatingly dull, and the characters lacked depth. The plot felt disjointed, and I found myself struggling to finish it. I would strongly advise against reading this Kindle book.",
    "One star is too generous for this book. It was a complete letdown. The story was poorly developed, and the writing was uninspiring. I couldn't find any redeeming qualities in this read.",
    "This Kindle book was a total disappointment. The characters were unrelatable, and the plot was uninteresting. It felt like a never-ending slog, and I regret spending any money on it.",
    "I had high hopes for this book, but it failed to deliver. The writing lacked finesse, and the plot was predictable. It was a frustrating read that left me thoroughly unsatisfied.",
    "I wish I could get a refund for this book. It was a regrettable purchase. The story was poorly constructed, and I couldn't connect with the characters. I was left underwhelmed and disappointed.",
    "This book was a major letdown. The characters were one-dimensional, and the plot was lackluster. I expected so much more, but it failed to meet my expectations.",
    "I was thoroughly unimpressed with this book. The writing style was uninspiring, and the story lacked depth. I struggled to find any enjoyment in this read.",
    "A forgettable story with no impact. The plot twists were uninspired, and the characters were forgettable. It was a disappointing experience from start to finish.",
    "I kept hoping it would get better, but it never did. The story was a constant disappointment, and I was left wondering why I wasted my time on this Kindle book.",
    "This book didn't live up to the hype. The writing was mediocre at best, and the plot failed to engage me. It was a forgettable story that left no impact.",

    "I expected more from this Kindle book, but it offered an average reading experience. The story was middle-of-the-road, and the characters didn't leave a lasting impression. It wasn't remarkable but wasn't terrible either.",
    "An underwhelming experience with no standout features. This book had the potential for more, but it failed to deliver consistently. It was an average read that didn't stand out but had some redeeming moments.",
    "A Kindle book that failed to impress fully. It was an unremarkable read with no lasting impact, though it had its moments of intrigue. I found it difficult to connect with the story and characters consistently.",
    "I wish I had spent my time on a better book, but this book had its moments. It didn't offer anything special or unique, but it had some redeeming qualities. It was an average story that left me somewhat indifferent.",
    "This book didn't live up to my expectations, although it had some moments of promise. I had hoped for more depth and complexity, but it fell short at times. The story was forgettable but not without merit.",
    "A predictable and unoriginal story, but it had some interesting moments. This book lacked the wow factor but had occasional flashes of brilliance. It was an average read that didn't leave a strong impression.",
    "I wouldn't recommend this book to others, but it had its share of intriguing elements. It was an average plot with no remarkable features, but it had moments that held my interest. I struggled to find any redeeming qualities in this read.",
    "I was left uninspired by this book, although it had its moments of intrigue. It was a forgettable story that failed to captivate my interest fully, but it had redeeming qualities that kept me reading.",
    "This book was far from what I had hoped for, but it had its moments of intrigue. It was an average Kindle book with no standout qualities, but it had some redeeming qualities that kept me reading.",
    "I expected more from this book, although it had its moments of intrigue. It was an average read with no remarkable plot twists, but it had redeeming qualities that made it worth the read. The characters were forgettable, and the writing style was uninspiring.",

    "A gripping page-turner from start to finish! This book had me on the edge of my seat, and I couldn't put it down. The characters were relatable, and the plot twists were captivating.",
    "This book changed my life. It's a must-read that left a lasting impact. The writing is simply beautiful, and the story touched my heart.",
    "The characters were so relatable; I couldn't get enough. This Kindle book transported me to another world, and I was completely immersed in the story.",
    "The plot twists kept me guessing until the very end. I laughed, I cried, and I couldn't have asked for a better book to read. It's a literary gem.",
    "An emotional rollercoaster that tugged at my heartstrings. This book is a masterfully crafted story that will stay with you long after you finish it.",
    "Couldn't stop reading, even in the wee hours of the morning. This Kindle book was a fantastic escape from reality, and I'm recommending it to all my friends.",
    "A literary gem in the world of Kindle books. I was completely immersed in the world the author created, and it was an incredible journey of self-discovery.",
    "This book was like a warm hug for the soul. It's an instant classic that I'll read again and again. A story that will touch your heart.",
    "I wish I could give it more than five stars. This book will keep you hooked from the first page. It's a true masterpiece of storytelling.",
    "This book was the highlight of my reading list. The characters were relatable, and the plot was filled with suspense. It was an absolute masterpiece.",
]


# Function to convert the time format
def convert_time_format(original_time):
    parsed_time = datetime.strptime(original_time, "%Y-%m-%dT%H:%M:%S.%f")
    return parsed_time.strftime("%Y-%m-%d %H:%M:%S")


def predict_comment(score):
    # Map prediction scores to comments
    comments = {
        1: "👎 Not a recommended book.",
        2: "😔 An average read with room for improvement.",
        3: "🙄 A good book worth considering.",
        4: "👍 A highly recommended read!",
        5: "💯 An outstanding book that you must read!"
    }

    # Display the comment based on the prediction score
    if score in comments:
        comment = comments[score]
        st.info(comment)  
    else:
        st.error("Invalid prediction score")


def predict_and_display(api_url, input):
    response = requests.post(url=api_url, json=input)

    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        if predictions:
            for prediction in predictions:
                st.success(f"Predicted Rating: {prediction['rating']}")
        else:
            st.warning("No predictions returned.")
    else:
        st.write("Prediction failed. Please check your input and try again.")

    predict_comment(prediction['rating'])    


def predict_batch(api_url, review_texts):
    input_data = [{"review": text, "predict_type": "App"} for text in review_texts]
    response = requests.post(url=api_url, json=input_data)

    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        return [prediction.get("rating", None) for prediction in predictions]
    else:
        return [None] * len(review_texts)


# Define a function to fetch predictions from the API
def get_predictions(api_url, start_date, end_date, start_time, end_time, selected_ratings, selected_types):
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "start_time": start_time,
        "end_time": end_time,
        "selected_ratings": selected_ratings,
        "selected_types": selected_types
    }

    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        predictions = response.json()
        return predictions
    else:
        st.error("Failed to fetch predictions from the API.")
        return None
