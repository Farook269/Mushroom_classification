# Mushroom Classification Using Machine Learning

This project predicts whether a mushroom is edible or poisonous based on its physical and ecological attributes. Using machine learning techniques, the project processes the dataset to build predictive models, aiding in quick and accurate classifications.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Technologies Used](#technologies-used)
5. [Setup and Installation](#setup-and-installation)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)

---

## Introduction

The project employs machine learning techniques to classify mushrooms as edible or poisonous. It analyzes key features like cap shape, odor, gill size, and habitat to make predictions, making it useful for educational and research purposes.

---

## Dataset

- **Source:** The dataset contains 23 features describing various properties of mushrooms, including cap shape, color, bruises, odor, gill characteristics, stalk properties, and more.

![Screenshot 2024-12-11 233529](https://github.com/user-attachments/assets/0f1b2c6b-412e-45c8-bc57-2d6e2e30a7e7)
- **Target Variable:** `class` (edible = `e`, poisonous = `p`)



- **Preprocessing:**
  - Handled categorical data by mapping to numerical values.
  - Scaled features for training.
  - Removed duplicate entries for consistency.
  ![Screenshot 2024-12-11 233730](https://github.com/user-attachments/assets/44287c8c-ca61-4b66-a2ea-3a68b0012f8f)


---

## Project Workflow

1. **Data Exploration**

   - Checked dataset shape, columns, and missing values.

      ![Screenshot 2024-12-11 234039](https://github.com/user-attachments/assets/9602af32-47c8-441c-bcde-5116e0e83ac1)

   - Visualized distributions and correlations between features.


   ![Screenshot 2024-12-11 234156](https://github.com/user-attachments/assets/9d4ebf96-4c30-4d52-aa70-75267a1f8277)

2. **Data Preprocessing**

   - Converted categorical features to numerical values using mapping.
   - Scaled data to prepare for machine learning models.

   ![Screenshot 2024-12-11 233856](https://github.com/user-attachments/assets/f2ea8277-650d-48a0-aff5-be1f8bb809c8)

3. **Model Training**

   - Trained various models, including:
     - Logistic Regression
     - Random Forest
     - K-Nearest Neighbors
   - Conducted hyperparameter tuning using GridSearchCV.

   ![Screenshot 2024-12-11 234326](https://github.com/user-attachments/assets/51bfaba8-b922-4785-9790-e4dd97dc6693)

4. **Model Evaluation**

   - Evaluated models using metrics like accuracy, confusion matrix, and classification reports.
   - Selected the best-performing model for deployment.


   

![Screenshot 2024-12-11 234250](https://github.com/user-attachments/assets/10bdfb6a-8f67-4c77-8231-bfc764b6311b)



5. **Deployment**

   - Saved the trained model using `joblib` for future predictions.
   - Created a GUI for user interaction.

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib
- **Tools:** Jupyter Notebook

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Farook269/Mushroom_classification.git
   
   cd CampusPlacementPrediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

   ```

3. Run the notebook:
   ```bash
   jupyter notebook Mushroom_Classification.ipynb

   ```

4. Launch the GUI:
   ```bash
   python gui.py
   ```

---

## Result 

Key Insights:

- Visualized feature importance and correlations.
- Trained and evaluated multiple machine learning models.
- Achieved high accuracy in predicting mushroom edibility.

Model performance:

- Random Forest performed best with an accuracy exceeding 95%.

- Confusion matrices and classification reports demonstrated robustness.








---

## Future Enhancements


- Add more detailed visualizations for feature importance.

- Expand dataset with additional features for better predictions.

- Deploy a web-based interface for real-time predictions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
