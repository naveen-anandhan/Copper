# Copper Price and Status Prediction App

Welcome to the Copper Price and Status Prediction App, a machine learning application designed to predict the selling price and status (Won/Lost) for copper products. This app leverages various machine learning models to aid the copper industry in making data-driven decisions regarding sales and pricing.

## Overview

This application provides two main functionalities:
1. **Regression**: Predicts the selling price of copper using the ExtraTreeRegressor model.
2. **Classification**: Predicts the status of copper deals (Won/Lost) using the RandomForestClassifier model.

## Features

- **Home Page**: Contains a detailed description of the problem statement, the tools and technologies used, and an overview of the machine learning models implemented.
- **Model Page**: Allows users to input various parameters and get predictions for copper price and status.

## Tools and Technologies Used

- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Pickle

## Getting Started

### Prerequisites

Make sure you have Python installed on your system. You can download it from the [official Python website](https://www.python.org/).

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/copper-prediction-app.git
   cd copper-prediction-app
   ```

2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage

### Home Page

The home page provides an overview of the project, including:
- Problem Statement
- Tools and Technologies Used
- Description of the Machine Learning Models (ExtraTreeRegressor for Regression and RandomForestClassifier for Classification)
- Author Information

### Model Page

On the model page, you can select between Regression and Classification tasks. 

#### Regression

1. Input the required parameters such as Quantity, Thickness, Width, Country Code, Status, Item Type, Application Type, Product Reference, Order Date, and Estimated Delivery Date.
2. Click the "PREDICT PRICE" button to get the predicted selling price of copper.

#### Classification

1. Input the required parameters such as Quantity, Thickness, Width, Selling Price, Country Code, Item Type, Application Type, Product Reference, Order Date, and Estimated Delivery Date.
2. Click the "PREDICT STATUS" button to get the predicted status (Won/Lost) of the copper deal.

## Project Structure

```
copper-prediction-app/
├── app.py                   # Main application file
├── requirements.txt         # Python packages required
├── models/                  # Directory containing the ML models
│   ├── et_reg.pkl           # ExtraTreeRegressor model for price prediction
│   ├── RF_class.pkl         # RandomForestClassifier model for status prediction
│   ├── country.pkl          # Encoder for country codes
│   ├── status.pkl           # Encoder for status
│   ├── item_type.pkl        # Encoder for item types
│   └── scaling.pkl          # Scaler for data normalization
└── README.md                # Project documentation
```

## Contributing

Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are warmly welcome.


## Contact

Created by [Naveen Anandhan](https://www.linkedin.com/in/naveen-anandhan-8b03b62a5/?trk=public-profile-join-page) - feel free to contact me!

