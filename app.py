import streamlit as st
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_model_and_scaler()

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn and explore insights from the dataset.")


with st.sidebar:
    st.header("🧾 Input Features")
    age = st.slider("Age", 18, 65, 30)
    tenure = st.slider("Tenure (months)", 1, 60, 10)
    usage = st.slider("Usage Frequency (months)", 0, 30, 15)
    support_calls = st.slider("Support Calls (last 30 days)", 0, 10, 5)
    payments = st.slider("Payment Delay", 0, 30, 10)
    totalcharges = st.slider("Total Charges ($)", 100, 1000, 500)
    interaction = st.slider("Interactions", 1, 30, 15)
    gender = st.radio("Gender", ['Male', 'Female'])
    subscription = st.selectbox("Subscription Type", ['Basic', 'Standard', 'Premium'])
    contract = st.selectbox("Contract Type", ['Quarterly', 'Monthly', 'Annual'])

# Encode categorical variables
genderselected = 0 if gender == 'Female' else 1
subscription_map = {'Basic': 0, 'Premium': 1, 'Standard': 2}
subscriptionselected = subscription_map[subscription]
contract_map = {'Annual': 0, 'Monthly': 1, 'Quarterly': 2}
contractselected = contract_map[contract]


st.divider()
st.subheader("🔮 Churn Prediction Result")

if st.button("Predict Churn"):
    input_data = np.array([[age, genderselected, tenure, usage, support_calls, payments,
                            subscriptionselected, contractselected, totalcharges, interaction]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    # if prediction == 1:
    #     result = "High chance of churn"
    #     # st.markdown("✅ **There is a high chance of churn**", unsafe_allow_html=True)
    # else:
    #     result = "Low chance of churn"
    #     # st.markdown("❌ **There is a low chance of churn**", unsafe_allow_html=True)

    # st.success(f"The model predicts: **{result}**")
    st.session_state['prediction_result'] = "High chance of churn" if prediction == 1 else "Low chance of churn"

    if 'prediction_result' in st.session_state:
        st.success(f"The model predicts: **{st.session_state['prediction_result']}**")

st.divider()
st.subheader("📈 Data Insights")

try:
    @st.cache_data
    def load_data():
        df = pd.read_csv("customer_churn_dataset-training-master.csv")
        df2 = pd.read_csv("preprocessed_data.csv")
        return df, df2

    df, df2 = load_data()


    with st.expander("📂 View Raw Data"):
        st.dataframe(df.head(30), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📊 Overview Charts", "📌 Advanced Visuals", "📉 Feature Correlation"])

    with tab1:
        st.markdown("### Distribution Overview")

        fig1 = px.histogram(df2, x='Age', color='Churn', barmode='overlay',
                                title='Age Distribution')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df2, x='Total Spend', nbins=50, title='Total Charges Distribution', color='Churn')
        st.plotly_chart(fig2, use_container_width=True)


        fig3 = px.pie(df2, names='Churn', title="Churn Breakdown (1 = Yes, 0 = No)")
        st.plotly_chart(fig3, use_container_width=True)
            
    
    with tab2:
        st.markdown("### Deeper Exploration")

        fig5 = px.scatter(df2, x='Total Spend', y='Tenure', color='Churn',
                          title='Total Spend vs Tenure by Churn')
        st.plotly_chart(fig5, use_container_width=True)

        grouped = df.groupby(['Gender', 'Subscription Type'])['Churn'].mean().reset_index()
        fig7 = px.density_heatmap(grouped, x='Gender', y='Subscription Type', z='Churn', 
                                  title='Churn Rate by Gender and Subscription Type', color_continuous_scale='Blues')
        st.plotly_chart(fig7, use_container_width=True)

        contract_churn = df.groupby('Contract Length')['Churn'].value_counts(normalize=True).unstack().fillna(0)
        fig8 = px.bar(contract_churn, barmode='group', title='Churn Rate by Contract Type',
                      labels={'value': 'Proportion'})
        st.plotly_chart(fig8, use_container_width=True)

    
    with tab3:
        st.markdown("### Correlation Overview")

        corr = df2.corr(numeric_only=True)

        st.markdown("#### 🔍 Interactive Correlation Heatmap")

        # Round correlation values for better readability
        rounded_corr = corr.round(2)

        # Create improved heatmap
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=rounded_corr.values,
                x=rounded_corr.columns,
                y=rounded_corr.index,
                colorscale="Viridis",
                zmin=-1,
                zmax=1,
                text=rounded_corr.values,
                texttemplate="%{text}",
                hovertemplate="Correlation between %{y} and %{x}: %{z}<extra></extra>",
                showscale=True
            )
        )

        fig_corr.update_layout(
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(tickangle=45),
            font=dict(size=12)
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### 📋 Top Correlations Table")
        corr_flat = corr.unstack().reset_index()
        corr_flat.columns = ['Feature 1', 'Feature 2', 'Correlation']
        corr_flat = corr_flat[corr_flat['Feature 1'] != corr_flat['Feature 2']]
        corr_flat['Abs Correlation'] = corr_flat['Correlation'].abs()
        corr_sorted = corr_flat.sort_values(by='Abs Correlation', ascending=False)
        st.dataframe(corr_sorted.head(30), use_container_width=True)

except FileNotFoundError:
    st.warning("⚠️ No dataset found for visualizations. Please ensure the required CSV files are in the working directory.")

st.caption("Made with ❤️ using Streamlit")

# Female --> 0 and Male --> 1
# Churn --> 1 and No Churn --> 0
