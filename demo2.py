import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to load dataset
def load_dataset():
    file_path = filedialog.askopenfilename()
    data = pd.read_csv(file_path)
    return data, file_path

# Function to preprocess data, train model, and save predictions
def process_and_save(data):
    # Preprocessing
    X = data['Context'] + " " + data['Question']  # Combine context and question
    y = data['Hallucination']  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    predictions = model.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)

    # Assign predictions to the dataset
    data['prediction'] = model.predict(tfidf_vectorizer.transform(X))

    # Save the dataset with predictions
    data.to_csv("dataset_with_predictions.csv", index=False)

    return accuracy

# Create GUI window
window = tk.Tk()
window.title("Dataset Processing")

# Function to handle button click
def on_click():
    data, file_path = load_dataset()
    accuracy = process_and_save(data)
    message_label.config(text=f"Dataset processed and predictions saved to dataset_with_predictions.csv file at {file_path}\nAccuracy: {accuracy:.4f}")

# Create load file button
button = tk.Button(window, text="Load Dataset File", command=on_click, bg="lightblue", font=("Arial", 12, "bold"))
button.pack(pady=10)

# Create label to display message
message_label = tk.Label(window, text="", font=("Arial", 12))
message_label.pack()

# Run the GUI
window.mainloop()
