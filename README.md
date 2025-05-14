# Customer Churn Prediction Dashboard

A Streamlit-based dashboard that predicts customer churn (i.e., whether a customer will leave a service) using a pre‑trained machine learning model, and provides interactive visual insights into customer behavior.

You can view the Churn Prediction App by clicking here: [Churn Prediction App] (https://churn-prediction-app-mzrr3tkq99fdh2h7yydtlq.streamlit.app/) 

---

## 🚀 Features

- **Interactive Prediction**: Enter customer features via a sidebar and get an immediate churn-risk prediction.
- **Data Exploration**: View raw and preprocessed datasets.
- **Overview Charts**: Histograms and pie charts of key features.
- **Advanced Visuals**: Scatter plots, heatmaps, and bar charts for deeper insights.
- **Correlation Analysis**: Interactive heatmap and top-correlations table.
- **Session Caching**: Fast loading of model, scaler, and datasets with Streamlit’s cache.

---

## 🧾 Input Features & Their Relevance to Churn

| Column              | Description                                                               | Relevance to Churn                                                                 |
|---------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| **Age**             | The age of the customer in years.                                         | Older or younger customers may show different loyalty or usage patterns.            |
| **Gender**          | Encoded as `0` (Female) and `1` (Male).                                   | Captures demographic trends; often less predictive but useful for segmentation.     |
| **Tenure**          | How many months the customer has been with the company.                   | Shorter tenure often correlates with higher churn (weaker relationship).            |
| **Usage Frequency** | Number of times the service is used per month.                            | Very low or very high usage can signal dissatisfaction or dependency.               |
| **Support Calls**   | Number of support calls made in the last 30 days.                         | More calls often indicate unresolved issues and frustration, raising churn risk.    |
| **Payment Delay**   | Average delay (in days) in making payments.                               | Consistent delays point to disengagement or financial difficulties.                 |
| **Subscription Type** | Service tier encoded as: <br>`0 = Basic`<br>`1 = Premium`<br>`2 = Standard` | Certain tiers may churn more if perceived value does not match cost.                |
| **Contract Length** | Contract type encoded as: <br>`0 = Annual`<br>`1 = Monthly`<br>`2 = Quarterly` | Longer contracts typically reduce churn due to customer commitment.                 |
| **Total Spend**     | Total amount of money spent by the customer.                              | Low spenders may be casual users; high spenders expect high satisfaction.           |
| **Last Interaction**| Days since the customer last used the service.                            | Longer inactivity suggests disengagement and higher churn probability.              |
| **Churn**           | Target variable: <br>`1 = Yes` (churned) <br>`0 = No` (retained)           | The outcome the model is trained to predict.                                        |

---

## 📁 Project Structure

- **`app.py`** / **`app_dupli.py`**  
  Streamlit application entry points.
- **`model.pkl`**  
  Serialized classification model for churn prediction.
- **`scaler.pkl`**  
  Fitted `StandardScaler` for input normalization.
- **`customer_churn_dataset-training-master.csv`**  
  Original training dataset.
- **`customer_churn_dataset-testing-master.csv`**  
  Original testing dataset.
- **`preprocessed_data.csv`**  
  Cleaned and preprocessed dataset used for visualizations.
- **`notebook.ipynb`**  
  Jupyter notebook detailing EDA, preprocessing, and model development.
- **`requirements.txt`**  
  Python dependencies.

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.7 or higher  
- Recommended: Create and activate a virtual environment  
  ```bash
  python -m venv venv
  source venv/bin/activate    # Linux/macOS
  venv\Scripts\activate       # Windows

### Installations
- Clone the repository
  ```bash
  git clone https://github.com/Jolvin-James/Churn-Prediction-App.git
  cd Churn-Prediction-App
- Install dependencies
  ```bash
  pip install -r requirements.txt

