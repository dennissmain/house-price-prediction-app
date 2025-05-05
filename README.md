🏡 House Price Prediction App

A comprehensive, MLOps-driven house price prediction platform built with Python, Streamlit, Docker, MongoDB, and Azure. This application offers real-time predictions, seamless scalability, and continuous model retraining, providing an end-to-end solution from exploratory modeling to a fully automated, cloud-deployed web app.

<img width="1512" alt="Screenshot 2025-05-05 at 19 25 19" src="https://github.com/user-attachments/assets/b09c9589-77f0-45e6-9fd8-be7807f26778" />




🚀 Features

	•	Interactive Web Interface: User-friendly Streamlit app for inputting property details and viewing predictions.
	•	Multiple ML Models: Incorporates CatBoost, XGBoost, and a deep learning model for robust predictions.
	•	Hybrid Ensemble Model: Combines multiple models to enhance prediction accuracy.
	•	MongoDB Integration: Stores user inputs and prediction results for analysis and model retraining.
	•	Dockerized Deployment: Ensures consistent environments across development and production.
	•	Azure Integration: Deployed on Azure for scalability and reliability.
	•	Continuous Retraining: Implements MLOps practices for model monitoring and retraining.

🧰 Tech Stack

	•	Frontend: Streamlit
	•	Backend: Python
	•	Machine Learning: scikit-learn, XGBoost, CatBoost, TensorFlow
	•	Database: MongoDB
	•	Containerization: Docker
	•	Cloud Platform: Azure
	•	CI/CD: GitHub Actions

📂 Project Structure

```mermaid
graph TD
    User -->|Inputs house data| StreamlitApp
    StreamlitApp -->|Calls model| PredictionEngine
    PredictionEngine -->|Returns prediction| StreamlitApp
    StreamlitApp -->|Saves input + result| MongoDB


📦 Installation

Prerequisites
	•	Python 3.7 or higher
	•	Docker (for containerized deployment)
	•	MongoDB instance (local or cloud-based)

Clone the Repository

git clone https://github.com/dennissmain/house-price-prediction-app.git
cd house-price-prediction-app


🤖: Want to try the app?  Run the Application -

https://house-price-prediction-app-v1.streamlit.app/

The application will be accessible at http://localhost:8501.


📊 Usage
	1.	Input Features: Enter property details such as location, size, number of bedrooms, etc.
	2.	View Prediction: The app displays the predicted house price along with model confidence.
	3.	Data Storage: All inputs and predictions are stored in MongoDB for future analysis and model retraining.

🗄️ MongoDB Integration

The application uses MongoDB to store user inputs and prediction results. This integration allows for:
	•	Data Persistence: Ensures that user interactions are saved for future reference.
	•	Model Retraining: Facilitates continuous learning by providing real-world data for model updates.
	•	Analytics: Enables analysis of user behavior and model performance over time.

🤝 Contributing

Contributions are welcome! Please follow these steps:
	1.	Fork the repository.
	2.	Create a new branch: git checkout -b feature/YourFeature
	3.	Commit your changes: git commit -m 'Add YourFeature'
	4.	Push to the branch: git push origin feature/YourFeature
	5.	Open a pull request.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact

For questions or suggestions, please contact okwechimedenniss@gmail.com.
