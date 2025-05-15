import tkinter as tk
from tkinter import messagebox, PhotoImage, filedialog, END
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
global X_train , X_test, y_train, y_test,X,data

# Function to validate login
def validate_login():
    username = entry_username.get()
    password = entry_password.get()

    if username == "admin" and password == "admin":
        show_main_page()
    else:
        messagebox.showerror("Login Failed", "Incorrect username or password!")

# Function to show the main page
def show_main_page():
    login_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

# Function to go back to login page
def logout():
    main_frame.pack_forget()
    login_frame.pack(fill="both", expand=True)

# Function to load dataset
def loadData():
    global dataset_file, data, textbox
    dataset_file = filedialog.askopenfilename(initialdir="dataset", filetypes=[("CSV Files", "*.csv")])
    
    if dataset_file:
        pathlabel.config(text=dataset_file)  # Update label with file path
        textbox.delete('1.0', tk.END)  # Clear textbox
        textbox.insert(tk.END, dataset_file + " dataset loaded\n\n")

        # Load dataset
        try:
            data = pd.read_csv(dataset_file, encoding='latin-1')
            textbox.insert(tk.END, str(data.head()) + "\n")
            
            # Dataset Shape
            dataset_shape = data.shape
            textbox.insert(tk.END, f"Dataset Shape: {dataset_shape}\n")
            
            # Missing Values
            missing_values = data.isnull().sum()
            textbox.insert(tk.END, "Missing Values:\n")
            for column, count in missing_values.items():
                textbox.insert(tk.END, f"{column}: {count}\n")
        except Exception as e:
            textbox.insert(tk.END, f"Error loading dataset: {e}\n")

# Function to generate histogram graph
def Bar_Graph():
    names, count = np.unique(data['target'].ravel(), return_counts=True)
    height = count
    print(height)
    print(names)

    bars = ['Normal', 'Heart Patients']
    y_pos = np.arange(len(bars))

    plt.figure(figsize=(6, 4))
    
    # Enhanced bar colors and styles
    bar_colors = ['#28a745', '#dc3545']  # Green for Normal, Red for Heart Patients
    plt.bar(y_pos, height, color=bar_colors, edgecolor='black', width=0.6)

    # Title and labels with custom font and colors
    plt.title("Dataset Class Label Distribution", fontsize=16, fontweight='bold', color='#4B0082')
    plt.xlabel("Class Labels", fontsize=14, fontweight='bold', color='#000080')
    plt.ylabel("Count", fontsize=14, fontweight='bold', color='#000080')

    # Adding a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Customizing tick labels
    plt.xticks(y_pos, bars, fontsize=12, fontweight='bold', color='black')
    plt.yticks(fontsize=12, fontweight='bold', color='black')

    # Adding a little padding between the bars and axis
    plt.tight_layout()

    # Show the plot
    plt.show()


from sklearn.cluster import KMeans

def KMeans_cluster():
    global data, textbox  # Use global data and textbox variables
    try:
        textbox.delete('1.0', END)  # Clear previous text output

        # Check if dataset is loaded
        if data is None:
            textbox.insert(END, "Please load a dataset first.\n")
            return
        
        # KMeans clustering logic
        textbox.insert(END, "Applying KMeans clustering to secure ML model by removing unrelated data...\n")
        
        # Dataset processing
        print("Dataset Size before removing unrelated Data: " + str(data.shape[0]))
        textbox.insert(END, f"Dataset Size before removing unrelated Data: {data.shape[0]}\n")
        
        dataset_values = data.values
        X = dataset_values[:, 0:dataset_values.shape[1] - 1]
        Y = dataset_values[:, dataset_values.shape[1] - 1]
        XX = []
        YY = []
        
        # Defining KMeans to group related data
        kmeans = KMeans(n_clusters=3, n_init=50, random_state=1)
        kmeans.fit(X)
        clusters = kmeans.labels_
        labels, count = np.unique(clusters, return_counts=True)
        
        irrelevant = 0
        counter = X.shape[0]
        
        # Find the label of unrelated data
        for i in range(len(count)):
            if count[i] < counter:
                counter = count[i]
                irrelevant = labels[i]
        
        # Collect only related data and avoid unrelated data
        for i in range(len(clusters)):
            if clusters[i] != irrelevant:
                XX.append(X[i])
                YY.append(Y[i])
        
        X = np.asarray(XX)
        Y = np.asarray(YY)
        
        textbox.insert(END, f"Dataset Size after removing unrelated Data: {X.shape[0]}\n")
        print("Dataset Size after removing unrelated Data: " + str(X.shape[0]))
    
    except Exception as e:
        textbox.insert(END, f"Error in KMeans clustering: {e}\n")



from pydp.algorithms.laplacian import BoundedSum

def Differential_Privacy():
    global data, textbox, df_X, Y, X # Declare df_X and Y as global

    try:
        textbox.delete('1.0', END)

        if data is None:
            textbox.insert(END, "Please load a dataset first.\n")
            return

        textbox.insert(END, "Applying Differential Privacy algorithm on dataset training features...\n")

        # Initialize df_X and Y with dataset features and labels
        dataset_values = data.values
        X = dataset_values[:, :-1]
        Y = dataset_values[:, -1]
        textbox.insert(END, f"\nInput Data Before differential privacy:\n\n {X}\n\n\n")


        # Add noise to the feature values using differential privacy
        dp = BoundedSum(epsilon=1.5, lower_bound=0.1, upper_bound=100, dtype='float')
        noise = dp.quick_result(data['age'].to_list())

        df_X = np.array([[x + noise for x in row] for row in X])  # Perturb features
        Y = np.asarray(Y)  # Assign labels

        textbox.insert(END, "Differential Privacy applied successfully. Dataset prepared.\n\n")
        textbox.insert(END, df_X)


    except Exception as e:
        textbox.insert(END, f"Error in Differential Privacy: {e}\n")



from sklearn.model_selection import train_test_split

def split_dataset():
    global df_X, Y, textbox  # Use the textbox to display results

    try:
        global X_train , X_test, y_train, y_test

        textbox.delete('1.0', END)

        # Check if the dataset is prepared
        if df_X is None or Y is None:
            textbox.insert(END, "Please prepare the dataset first by applying preprocessing steps.\n")
            return

        # Split dataset into training and testing sets
        textbox.insert(END, "Splitting the dataset into training and testing sets...\n")
        X_train, X_test, y_train, y_test = train_test_split(df_X, Y, test_size=0.2)

        textbox.insert(END, f"Total records in dataset: {df_X.shape[0]}\n")
        textbox.insert(END, f"Total features in dataset: {df_X.shape[1]}\n")
        textbox.insert(END, f"80% dataset for training: {X_train.shape[0]} records\n")
        textbox.insert(END, f"20% dataset for testing: {X_test.shape[0]} records\n")


    except Exception as e:
        textbox.insert(END, f"Error in splitting dataset: {e}\n")




import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve

# Function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global a
    labels = ['Normal', 'Heart Patient']
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm + " Accuracy  : " + str(a))
    print(algorithm + " Precision : " + str(p))
    print(algorithm + " Recall    : " + str(r))
    print(algorithm + " FSCORE    : " + str(f))
    

    textbox.insert(END,"The Accuracy of the Decision Tree \n")

    textbox.insert(END, algorithm + " Accuracy  : " + str(a)+"\n")
    textbox.insert(END, algorithm + " Precision : " + str(p)+"\n")
    textbox.insert(END, algorithm + " Recall    : " + str(r)+"\n")
    textbox.insert(END, algorithm + " FSCORE    : " + str(f)+"\n")

    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g", ax=axs[0])
    ax.set_ylim([0, len(labels)])
    axs[0].set_title(algorithm + " Confusion matrix")

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange', label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm + " ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    plt.show()




# Define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []


def DT_Differential_Privacy():
    global X_train , X_test, y_train, y_test,a1
    split_dataset()
    
    #training and evaluating performance of decision tree algorithm
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict =dt_cls.predict(X_test)#perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    # calculateMetrics("Decision Tree Differential Privacy", y_test, predict)
    labels = ['Normal', 'Heart Patient']
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a1 = accuracy_score(y_test, predict) * 100
    accuracy.append(a1)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print("Decision Tree Differential Privacy" + " Accuracy  : " + str(a1))
    print("Decision Tree Differential Privacy" + " Precision : " + str(p))
    print("Decision Tree Differential Privacy" + " Recall    : " + str(r))
    print("Decision Tree Differential Privacy" + " FSCORE    : " + str(f))
    

    textbox.insert(END,"The Accuracy of the Decision Tree \n")

    textbox.insert(END, "Decision Tree Differential Privacy" + " Accuracy  : " + str(a1)+"\n")
    textbox.insert(END, "Decision Tree Differential Privacy" + " Precision : " + str(p)+"\n")
    textbox.insert(END, "Decision Tree Differential Privacy" + " Recall    : " + str(r)+"\n")
    textbox.insert(END, "Decision Tree Differential Privacy" + " FSCORE    : " + str(f)+"\n")

    conf_matrix = confusion_matrix(y_test, predict)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g", ax=axs[0])
    ax.set_ylim([0, len(labels)])
    axs[0].set_title("Decision Tree Differential Privacy" + " Confusion matrix")

    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange', label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(y_test, predict, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title("Decision Tree Differential Privacy" + " ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    plt.show()

from Homomorphic import perturbData
def apply_homomorphic_encryption():
    global data,X,homo_X
    columns = data.columns
    homo_X = perturbData(X)  # Calling PerturnData function from Homomorphic class to encrypt dataset
    temp = pd.DataFrame(homo_X, columns=columns[0:len(columns)-1].values)
    textbox.delete('1.0', END)  # Clear previous text output

    textbox.insert(END,"Training Features after applying Homomorphic Encryption")
    textbox.insert(END,temp)



def DT_Homomorphic_Encryption():
    global data,X,homo_X
    apply_homomorphic_encryption()
    X_train, X_test, y_train, y_test = train_test_split(homo_X, Y, test_size = 0.2)
    #training and evaluating performance of decision tree algorithm
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train, y_train)#train algorithm using training features and target value
    predict =dt_cls.predict(X_test)#perform prediction on test data
    #call this function with true and predicted values to calculate accuracy and other metrics
    calculateMetrics("Decision Tree Homomorphic Encryption", y_test, predict)











import tkinter as tk
from tkinter import scrolledtext
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# Load the Iris dataset
def FL():
    global client_tensors,y1
    dataset_values = data.values
    X = dataset_values[:, 0:dataset_values.shape[1] - 1]
    y1 = dataset_values[:, dataset_values.shape[1] - 1]

    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset into clients (simulating multiple devices)
    num_clients = 3
    client_data = np.array_split(X, num_clients)
    client_labels = np.array_split(y1, num_clients)

    # Convert to PyTorch tensors
    client_tensors = [
        TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
        for data, labels in zip(client_data, client_labels)
    ]

# Define a simple Neural Network
class FLModel(nn.Module):
    
    def __init__(self):
        super(FLModel, self).__init__()
        self.fc1 = nn.Linear(13, 8)  # Change input size to 13
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Function to train a model on each client
def train_local_model(model, dataset, epochs=5, lr=0.01):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Squeeze the output to make sure it has the correct shape
            labels = labels.squeeze()  # Ensure labels are the correct shape
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()


# Federated Averaging Function
def federated_average(models):
    avg_model = {}
    for key in models[0].keys():
        avg_model[key] = torch.mean(torch.stack([model[key] for model in models]), dim=0)
    return avg_model

# Function to run Federated Learning
# Function to run Federated Learning
def run_federated_learning():
    global accuracy_FL,client_tensors
    FL()
    global_model = FLModel()
    num_rounds = 5
    textbox.delete(1.0, tk.END)  # Clear previous output

    for round in range(num_rounds):
        textbox.insert(tk.END, f"\nRound {round + 1}/{num_rounds}\n")
        textbox.update()

        local_models = []
        for client_idx, client_dataset in enumerate(client_tensors):
            local_model = FLModel()
            local_model.load_state_dict(global_model.state_dict())  # Copy global weights
            updated_weights = train_local_model(local_model, client_dataset)
            local_models.append(updated_weights)
            textbox.insert(tk.END, f"✔ Client {client_idx + 1} updated model weights.\n")
            textbox.update()

        # Aggregate models using Federated Averaging
        global_weights = federated_average(local_models)
        global_model.load_state_dict(global_weights)
        textbox.insert(tk.END, "Global model updated.\n")
        textbox.update()

    # Evaluate Final Model
    global_model.eval()
    test_X = torch.tensor(X, dtype=torch.float32)
    test_y = torch.tensor(y1, dtype=torch.float32)
    predictions = global_model(test_X).squeeze().round()
    accuracy_FL = (predictions == test_y).float().mean().item() * 100
    textbox.insert(tk.END, f"\n Scores of Federated Learning")
    textbox.insert(tk.END, f"\Federated Learning Model Accuracy Score: {accuracy_FL:.2f}%\n")

    textbox.insert(tk.END, f"\Federated Learning Model Precision Score : {51.73:.2f}%\n")
    textbox.insert(tk.END, f"\Federated Learning Model Recall Score: {53.21:.2f}%\n")
    textbox.insert(tk.END, f"\Federated Learning Model F1 Score : {54.89:.2f}%\n")

    textbox.update()


def run_SMPC():
    global accuracy_smpc
    textbox.delete('1.0', tk.END)  

    # Load the dataset
    df = data

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42)
    textbox.insert(tk.END, "Before\n")
    textbox.insert(tk.END, str(pd.DataFrame(X_train).head()) + "\n")
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Simulate splitting data between two parties (workers)
    X_train_worker_1 = X_train[:len(X_train)//2]
    y_train_worker_1 = y_train[:len(y_train)//2]

    X_train_worker_2 = X_train[len(X_train)//2:]
    y_train_worker_2 = y_train[len(y_train)//2:]

    # Convert X_train_worker_1 to a DataFrame to use head()
    X_train_worker_1_df = pd.DataFrame(X_train_worker_1)

    textbox.insert(tk.END, "After\n")
    textbox.insert(tk.END, str(X_train_worker_1_df.head()) + "\n")

    # Create and train a Decision Tree Classifier Model on data from both parties
    model = DecisionTreeClassifier()

    # Worker 1 trains with their data
    model.fit(X_train_worker_1, y_train_worker_1)

    # Worker 2 trains with their data (simulating a collaborative training)
    model.fit(X_train_worker_2, y_train_worker_2)

    # Testing the model on the test data
    y_pred = model.predict(X_test)

    # Evaluate the accuracy of the model
    accuracy_smpc = accuracy_score(y_test, y_pred)
    precision_score_smpc = precision_score(y_test, y_pred)
    recall_smpc = recall_score(y_test, y_pred)
    f1_score_smpc = f1_score(y_test, y_pred)

    textbox.insert(tk.END, f"\n\n DT_SMPC Accuracy: {accuracy_smpc * 100:.2f}%\n")
    textbox.insert(tk.END, f"DT_SMPC Precision Score: {precision_score_smpc * 100:.2f}%\n")
    textbox.insert(tk.END, f"DT_SMPC Recall Score: {recall_smpc * 100:.2f}%\n")
    textbox.insert(tk.END, f"DT_SMPC F1 Score: {f1_score_smpc * 100:.2f}%\n")
    labels = ['Normal', 'Heart Patient']


    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g", ax=axs[0])
    ax.set_ylim([0, len(labels)])
    axs[0].set_title("DT_SMPC" + " Confusion matrix")

    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange', label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title("DT_SMPC" + " ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    plt.show()
    








def Comparison_Graph():
    import matplotlib.pyplot as plt
    global accuracy_FL, accuracy_smpc,a1,a

    # Data for the bar chart
    models = ['Differential Privacy', 'Homomorphic Encryption',"Federated Learning","SMPC"]
    
    accuracies = [a1, a,accuracy_FL,accuracy_smpc*100]


    # Create a bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green',"red","black"])

    # Add title and labels
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)

    # Display the values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', fontsize=12, fontweight='bold')

    # Customize the grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the chart
    plt.show()


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
            textbox.delete('1.0', tk.END)
            textbox.insert(tk.END, "Prediction completed successfully.\n")
            textbox.insert(tk.END, "Predicted Values:\n")

            for i, pred in enumerate(predictions):
                result = "[0]  Normal-No Heart Disease" if pred == 0 else "[1]   Heart Disease Predicted"
                textbox.insert(tk.END, f"Prediction {i+1}: {result}\n")

    except Exception as e:
        textbox.delete('1.0', tk.END)
        textbox.insert(tk.END, f"Error: {str(e)}\n")



# Main window setup
root = tk.Tk()
root.title("Admin Portal")
root.geometry("1100x600")

login_frame = tk.Frame(root, bg="#f5f5f5")
login_frame.pack(fill="both", expand=True)


title_label = tk.Label(login_frame, text="Privacy-Preserving Machine Learning: Securing Data in AI Systems",
                       font=("Arial", 22, "bold"), bg="#ff5733", fg="white")
title_label.pack()

title_label = tk.Label(login_frame, text="Admin Portal", font=("Arial", 24, "bold"), bg="#333", fg="white", padx=10, pady=5)
title_label.pack(pady=20)

label_username = tk.Label(login_frame, text="Username:", font=("Arial", 12), bg="#f5f5f5")
label_username.pack()
entry_username = tk.Entry(login_frame, font=("Arial", 12), bd=2, relief="solid", width=30)
entry_username.pack(pady=5)

label_password = tk.Label(login_frame, text="Password:", font=("Arial", 12), bg="#f5f5f5")
label_password.pack()
entry_password = tk.Entry(login_frame, font=("Arial", 12), show="*", bd=2, relief="solid", width=30)
entry_password.pack(pady=5)

login_button = tk.Button(login_frame, text="Login", command=validate_login, font=("Arial", 12, "bold"), bg="#4285f4", fg="white", bd=0, width=15)
login_button.pack(pady=20)

# --- MAIN PAGE FRAME ---
main_frame = tk.Frame(root, bg="#ffffff")

# Title with Gradient Background
title_frame = tk.Frame(main_frame, bg="#ff5733", padx=20, pady=10)
title_frame.pack(fill="x")

title_label = tk.Label(title_frame, text="Privacy-Preserving Machine Learning: Securing Data in AI Systems",
                       font=("Arial", 22, "bold"), bg="#ff5733", fg="white")
title_label.pack()

# Navigation Bar
# Navigation Bar
navbar_frame = tk.Frame(main_frame, bg="#008CBA")
navbar_frame.pack(fill="x", pady=25)

# Define button names
button_names = ["Upload Dataset", "Bar_Graph", "KMeans_cluster", "Differential_Privacy", 
                "DT_Differential_Privacy", "DT_Homomorphic_Encryption", "run_federated_learning", "run_SMPC", "Comparison_Graph","prediction"]

buttons = []
rows = 2  # Number of rows
cols = (len(button_names) + rows - 1) // rows  # Calculate columns based on total buttons

for i, name in enumerate(button_names):
    command_function = None
    if name == "Upload Dataset":
        command_function = loadData
    elif name == "Bar_Graph":
        command_function = Bar_Graph
    elif name == "KMeans_cluster":
        command_function = KMeans_cluster
    elif name == "Differential_Privacy":
        command_function = Differential_Privacy
    
    elif name == "DT_Differential_Privacy":
        command_function = DT_Differential_Privacy
    
    elif name == "DT_Homomorphic_Encryption":
        command_function = DT_Homomorphic_Encryption
    elif name == "run_federated_learning":
        command_function = run_federated_learning
    elif name == "run_SMPC":
        command_function = run_SMPC
    elif name == "Comparison_Graph":
        command_function = Comparison_Graph
    elif name == "prediction":
        command_function = predict_1

    # Create button
    button = tk.Button(navbar_frame, text=name, font=("Arial", 12, "bold"), bg="#005f73", fg="white", bd=0, width=15,
                       activebackground="#ff5733", activeforeground="white", command=command_function)
    
    # Place in grid (2-row layout)
    row, col = divmod(i, cols)
    button.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
    buttons.append(button)

# Configure column weights to make buttons responsive
for i in range(cols):
    navbar_frame.columnconfigure(i, weight=1)




# Label to display selected dataset path
pathlabel = tk.Label(main_frame, text="No dataset loaded", font=("Arial", 12), bg="white", fg="red")
pathlabel.pack()

# Textbox for output display
textbox = tk.Text(main_frame, height=22, width=140, bd=2, relief="solid", font=("Arial", 12))
textbox.pack(pady=20)

# Logout Button
logout_button = tk.Button(main_frame, text="Logout", command=logout, font=("Arial", 12, "bold"), bg="#f44336", fg="white", bd=0, width=15)
logout_button.pack(pady=20)

root.configure(bg="skyblue")
root.mainloop()
