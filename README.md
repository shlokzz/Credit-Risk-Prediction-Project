# Credit Risk Prediction System

Built an end to end credit risk classifier which predicts whether a loan applicant will default using a Random Forest Classifier trained on 32000+ records - achieving around **93.3% accuracy** with around **0.97 precision**, deployed as an interactive web application.

[screenshot]

## The Problem
Banks need to identify high-risk loan applicants before approving credit. Missed defaults cost money; wrongly rejecting good applicants costs customers. This model optimizes for precision - minimizing false rejections while catching the majority of real defaults.

## Result
| Metric    | Score |
|-----------|-------|
| Accuracy  | 93.3% |
| Precision | 0.97  |
| Recall    | 0.7   |
| F1 Score  | 0.82  |

Precision was prioritized over recall to minimize false rejections  of credit worthy applicants 

## Key Decisions
- **Random Foreset Classifier** over **Logistic Regression** - Random Forest outperformed Logistic Regression by 3%  in cross-validation and handles non-linear relationships in financial data better.
- **Median imputation** - income and interest rate distributions are right-skewed; median  is more representative than mean.
- **Regularization by using max_depth and min_sample_leaf** - reduced a 6% train.test accuracy gap (overfitting) down to 0.4%.

## Tech Stack
Python · Pandas · Scikit-learn · Streamlit · Pickle

## Quick Start
```bash
git clone https://github.com/shlokzz/Credit-Risk-Prediction-Project.git
pip install -r requirements.txt
streamlit run apps/app.py
```

## Project Structure

```
notebooks/     # data cleaning, eda and model evaluation 
models/        # saved models and preprocessors
dataset/       # raw and cleaned data
apps/          # Streamlit web application 
```