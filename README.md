# Holiday Package Purchase Prediction

A machine learning project that predicts whether a customer will buy a holiday package, built with a Random Forest classifier and a Streamlit app.

Live app: [add your Streamlit link here]

## Why I made this

The dataset is from a fictional travel company, Trips & Travel. They want to launch a new Wellness Tourism package, but until now they contacted customers at random, which made marketing expensive. Only about 18% of customers ended up buying a package.

I wanted to see if customer details (age, income, occupation, how the sales pitch went, etc.) can tell us who is more likely to buy, so the sales team can call the promising customers first.

## Dataset

- Source: [Kaggle - Holiday Package Purchase Prediction](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)
- 4,888 customers and 20 columns
- Target column: `ProdTaken` (1 = bought the package, 0 = did not)
- The data is imbalanced: 920 buyers vs 3,968 non-buyers

## What I did

**1. Cleaning and preprocessing** (`notebooks/01_EDA_and_Preprocessing.ipynb`)
- Fixed inconsistent labels (`Fe Male` to `Female`, `Single` merged into `Unmarried`)
- Filled missing values with the median for numeric columns and the mode for categorical/discrete ones
- Dropped `CustomerID`
- Combined `NumberOfPersonVisiting` and `NumberOfChildrenVisiting` into one column, `TotalVisiting`

**2. Model training** (`notebooks/02_Model_Training_and_Evaluation.ipynb`)
- One-hot encoding for categorical columns, standard scaling for numeric ones
- Compared four models: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting
- Random Forest gave the best test accuracy, so I tuned it with `RandomizedSearchCV` and saved the final model with pickle

**3. Streamlit app** (`app.py`)
- Overview of the problem and the key numbers
- Data insights: purchase rate by group, income and age distributions
- Model performance: metrics, confusion matrix, ROC curve and feature importances (all on the test set)
- Predict tab: enter one customer's details and get a purchase probability, or upload a CSV to score many customers and download the ranked results
- A threshold slider in the sidebar to change how strict the model is

## Results

On the test set (20% of the data), at the default 0.5 threshold:

| Metric | Score |
|---|---|
| Accuracy | 93.05% |
| Precision (buyers) | 0.97 |
| Recall (buyers) | 0.66 |

The model is very precise: when it says someone will buy, it's usually right. But it misses about a third of the real buyers. For a marketing use case I'd lower the threshold to catch more of them, and the slider in the app shows how that trade-off works.

ROC-AUC: 0.981

## Limitations

- The data is small, and the model reaches 100% on the training set, so it overfits a bit even though the test scores are good.
- I tuned the model for accuracy, not recall, which is probably not the best choice for an imbalanced problem.
- Missing values were filled before the train/test split, which can leak a little information from the test data.
- Some columns, like pitch duration and number of follow-ups, are only known after a customer has already been contacted. In real use they might not be available when you want to decide who to call.
- The ROC plot saved in the notebook (`notebooks/auc.png`) has a hard-coded AUC label that doesn't match the curve. The app draws its own ROC curve with the correct value.

## Project structure

```
04_Random-Forest-Classifier/
├── data/
│   ├── Travel.csv
│   └── Travel_Cleaned.csv
├── models/
│   └── random_forest_model.pkl
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training_and_Evaluation.ipynb
│   └── auc.png
├── app.py
├── requirements.txt
└── README.md
```

## How to run it

```bash
git clone [https://github.com/CreatorSubodh/holiday-package-prediciton]
cd 04_Random-Forest-Classifier
pip install -r requirements.txt
streamlit run app.py
```

The saved model was trained with scikit-learn [your version], so use the same version from `requirements.txt` or the model may not load.

## Tools used

Python, pandas, NumPy, scikit-learn, matplotlib, Streamlit

## Things I'd like to improve

- Put the preprocessing and model into a single scikit-learn Pipeline
- Tune for recall or F1 instead of accuracy
- Use a stratified train/test split
- Try removing the columns that are only known after contact and see how much the score drops

## Author

Name : Subodh Kumar
GitHub link: https://github.com/CreatorSubodh
LinkedIn link: https://www.linkedin.com/in/subodh-kumar-993242230/?isSelfProfile=true