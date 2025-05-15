import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Function to train the model and display accuracy
def predict_1():
    try:
        # Load the dataset
        df = pd.read_csv("heart.csv")

        # Separate features and target variable
        x = df.drop(columns=['target'])  # Drop only the target column
        y = df['target']

        # Split dataset into training and test sets (80% train, 20% test)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        # Initialize DecisionTreeClassifier
        clf = DecisionTreeClassifier()

        # Train the model using the training sets
        clf.fit(x_train, y_train)

        # Make predictions
        y_pred = clf.predict(x_test)


        # Open file dialog to select the CSV file
        predict_file = filedialog.askopenfilename(title="Select a CSV File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if predict_file:
            # Load the new input data
            input_df = pd.read_csv(predict_file)

            # Replace '?' with NaN
            input_df.replace('?', np.nan, inplace=True)

            # Fill missing values with mode
            input_df.fillna(input_df.mode().iloc[0], inplace=True)

            # Predict using the trained Decision Tree model
            predictions = clf.predict(input_df)

            # Clear text box and insert predictions
            text_box.delete('1.0', tk.END)
            text_box.insert(tk.END, "Prediction completed successfully.\n")
            text_box.insert(tk.END, "Predicted Values:\n")
            for i, pred in enumerate(predictions):
                text_box.insert(tk.END, f"Prediction {i+1}: {pred}\n")

    except Exception as e:
        text_box.delete('1.0', tk.END)
        text_box.insert(tk.END, f"Error: {str(e)}\n")

# Create the Tkinter window
root = tk.Tk()
root.title("Decision Tree Classifier")
root.geometry("500x400")

# Add a button to run the model
train_button = tk.Button(root, text="Train Model", command=train_model, font=("Arial", 12), bg="lightblue", padx=10, pady=5)
train_button.pack(pady=10)

# Label to display results
result_label = tk.Label(root, text="Click 'Train Model' to start", font=("Arial", 12))
result_label.pack(pady=5)

# Add a text box to display predictions
text_box = tk.Text(root, height=10, width=60)
text_box.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
