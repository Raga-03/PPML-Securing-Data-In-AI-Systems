# 🛡️ PPML-Securing-Data-In-AI-Systems

## Overview
This project focuses on building a privacy-preserving machine learning framework to protect sensitive data—specifically health-related data—while enabling effective AI model training. We used a heart disease dataset to implement and demonstrate the practical application of privacy-enhancing techniques in real-world AI systems.

## Objective
To design and develop a secure ML system using:

Federated Learning – For decentralized model training without data sharing

Differential Privacy – To mask individual-level information

Homomorphic Encryption – To perform computations on encrypted data

Secure Multi-party Computation (SMPC) – For collaborative model building without data leaks

## Key Features
Preprocessing and clustering using K-Means to clean and optimize data

Added noise to sensitive features to ensure differential privacy

Trained models locally across nodes with federated learning

Encrypted input features with homomorphic encryption to ensure secure processing

Evaluated model using metrics like accuracy, precision, recall, and F1-score

Built user-friendly UI with secure login and dataset upload modules

## Technologies & Tools
Python 3.7+

Libraries: NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow Privacy, PyCryptodome, CustomTkinter

IDE: Jupyter Notebook, VS Code

## Use Case
This architecture is ideal for domains that handle sensitive user data such as:

Healthcare (e.g., disease prediction)

Finance (e.g., fraud detection)

Telecom (e.g., user analytics without data exposure)

## Outcome
The project demonstrated that it is feasible to train accurate ML models while maintaining high standards of data privacy. Our system aligns with privacy regulations (e.g., GDPR, HIPAA) and sets a foundation for ethical AI development.
