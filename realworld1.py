import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# Function to load dataset
def load_dataset():
    file_path = filedialog.askopenfilename()
    if file_path:  # Whether file is selected
        data = pd.read_csv(file_path)
        print("Dataset loaded. Shape:", data.shape)
        print("First few rows of the dataset:\n", data.head())
        return data, file_path
    else:
        return None, None  # If no file, return None

# Function to preprocess data, train model, and save predictions
def process_and_save(data, show_graph):
    # Preprocessing
    questions = data['Question'].astype(str)  # Convert to string
    answers = data['Answer'].astype(str)  # Convert to string

    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
    y = data['Hallucination']  # Ensure it's in integer format

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Plot ROC curve if required
    if show_graph:
        predictions_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, predictions_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.show()

    return accuracy, precision, recall, f1

# Create GUI window
window = tk.Tk()
window.title("Dataset Processing")

# Function to handle button click
def on_click():
    data, file_path = load_dataset()
    if data is not None:
        message_label.config(text="Processing...")
        accuracy, precision, recall, f1 = process_and_save(data, show_graph_var.get())
        message_label.config(text=f"Dataset processed\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
    else:
        message_label.config(text="No file selected!")

# Create load file button
button = tk.Button(window, text="Load Dataset File", command=on_click, bg="lightblue", font=("Arial", 12, "bold"))
button.pack(pady=10)

# Create checkbox to show graph
show_graph_var = tk.BooleanVar()
show_graph_check = tk.Checkbutton(window, text="Show ROC Curve", variable=show_graph_var)
show_graph_check.pack()

# Create label to display message
message_label = tk.Label(window, text="", font=("Arial", 12))
message_label.pack()

# Run the GUI
window.mainloop()
