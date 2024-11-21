# Machine Learning Project

## Project Overview

This project is a machine learning application designed to preprocess data, train various models, and visualize the results. It includes functionalities for data preprocessing, model training, evaluation, and visualization using Streamlit.

The project includes:
- Classification on wine data
- Regression on diabetes data
- Image processing on nail images

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/GaelBuenoBarthe/projet_machine_learning.git
    cd projet
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

4. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project is organized as follows:

```bash
├── data
│   ├── diabete.csv                # Dataset file for regression
│   └── wine.csv                   # Dataset file for classification
├── sections
│   ├── regression                    
│   │   ├── data_management        # Directory containing data_management.py
│   │   ├── data_preprocessing     # Directory containing data_preprocessing.py 
│   │   ├── data_visualization     # Directory containing data_visualization.py
│   │   ├── model                  # Directory containing model.py & neural_network.py
│   │   └── regression.py          # Regression model training,evaluation & management of streamlit app
│   ├── classification
│   │   ├── classification_visualization   # Directory containing all the visualizations for classification
│   │   ├── data_preprocessing     # Directory containing the preprocessing of the data
│   │   ├── models                 # Directory containing the models
│   │   └── classification.py      # Classification model training, evaluation evaluation & management of streamlit app
│   └── nailsdetection
│   │   ├── font                   # Directory containing font files
│   │   ├── pictures               # Directory containing nail images
│   │   └── nails.py               #  Visual detection model training,evaluation & management of streamlit app
├── app.py                         # App to run the Streamlit application
├── .gitignore                     # Git ignore file
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```
## Usage

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```
This will start the Streamlit server and open the application in your default web browser. 

## Features

* **Data Preprocessing**: Load and preprocess data, including handling missing values, removing duplicates, and dropping unnecessary columns.
* **Model Training**: Train various machine learning models and evaluate their performance.
* **Visualization**: Visualize data and model results using Streamlit and Matplotlib.

## Contributors

- Gael Bueno Barthe
- Marwa Benyahia
- Fabrice Mazenc
## Contributing

If you would like to contribute to this project, please follow these steps:  
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -m 'Add new feature').
4.  Push to the branch (git push origin feature-branch).
5.  Create a new Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.