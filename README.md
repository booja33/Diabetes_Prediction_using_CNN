
Diabetes Prediction using CNN
This repository contains code for a Convolutional Neural Network (CNN) model to predict diabetes based on various features. The model is trained using Python and Keras, and a Flask web application is provided for making predictions.

Contents
model_training.py: Python script to train the CNN model.
app.py: Flask web application for making predictions.
index.html: HTML file for the user interface of the web application.
diabetes.csv: Dataset used for training the model.
Setup
To run the web application and make predictions, follow these steps:

Clone this repository to your local machine:

bash
Copy code
git clone <repository-url>
Install the required dependencies. You can use pip to install them:

bash
Copy code
pip install flask numpy pandas scikit-learn keras
Run the Flask application:

bash
Copy code
python app.py
Open your web browser and navigate to http://127.0.0.1:5000 to use the application.

Usage
Once the web application is running, you can input values for various features related to diabetes (e.g., pregnancies, glucose level, blood pressure, etc.) in the provided form fields. After submitting the form, the application will make a prediction using the trained CNN model and display the result along with the probability of having diabetes.

Model Training
The CNN model is trained using the model_training.py script. It loads the dataset (diabetes.csv), preprocesses the data, builds the CNN model architecture, trains the model, and saves the trained model to a file (diabetes_cnn_model.h5).

Contributions
Contributions to improve the model's accuracy, the web application's interface, or any other aspect of the project are welcome. If you have any suggestions or find any issues, feel free to open an issue or pull request on GitHub.
