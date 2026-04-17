import pandas as pd
import numpy as np
import streamlit as st
from faker import Faker
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime

# --- DATA GENERATION ENGINE ---
def create_synthetic_finance_data(records=2000):
    fake = Faker()
    Faker.seed(101)
    np.random.seed(101)

    categories = {
        'Housing': 0.30,
        'Food & Dining': 0.25,
        'Transport': 0.15,
        'Entertainment': 0.10,
        'Healthcare': 0.05,
        'Utilities': 0.05,
        'Shopping': 0.10
    }

    pay_methods = ['Cash', 'UPI', 'Debit Card', 'Credit Card']
    data_list = []

    for _ in range(records):
        cat = np.random.choice(list(categories.keys()), p=list(categories.values()))

        if cat == 'Housing':
            amt = np.random.uniform(800, 2000)
        elif cat == 'Healthcare':
            amt = np.random.uniform(100, 1500)
        elif cat == 'Food & Dining':
            amt = np.random.uniform(5, 150)
        else:
            amt = np.random.uniform(10, 500)

        data_list.append({
            'Transaction_ID': fake.uuid4(),
            'Date': fake.date_between(start_date='-1y', end_date='today'),
            'Category': cat,
            'Amount': round(amt, 2),
            'Payment_Method': np.random.choice(pay_methods),
            'Merchant': fake.company(),
            'Note': fake.sentence(nb_words=4)
        })

    df = pd.DataFrame(data_list)
    df['Date'] = pd.to_datetime(df['Date'])

    anomaly_entry = pd.DataFrame([{
        'Transaction_ID': 'ANOMALY-001',
        'Date': datetime.now(),
        'Category': 'Food & Dining',
        'Amount': 5500.00,
        'Payment_Method': 'Credit Card',
        'Merchant': 'Luxury Catering Service',
        'Note': 'Corporate Event (Simulated)'
    }])

    df = pd.concat([df, anomaly_entry], ignore_index=True)

    return df.sort_values(by='Date').reset_index(drop=True)


# --- ANALYTICAL ENGINE ---
def perform_anomaly_detection(df):
    model = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly_Flag'] = model.fit_predict(df[['Amount']].values)
    df['Anomaly'] = df['Anomaly_Flag'].apply(
        lambda x: 'Anomalous' if x == -1 else 'Normal'
    )
    return df


def calculate_financial_kpis(df):
    total_spend = df['Amount'].sum()
    monthly_avg = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().mean()
    highest_cat = df.groupby('Category')['Amount'].sum().idxmax()
    return total_spend, monthly_avg, highest_cat


# --- DASHBOARD UI ---
def run_app():
    st.set_page_config(page_title="FinData Pro", layout="wide")
    st.title("💰 FinData Pro: Intelligent Expense Tracker")
    st.markdown("---")

    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = create_synthetic_finance_data()

    df = st.session_state.raw_df.copy()
    df = perform_anomaly_detection(df)

    # Sidebar
    st.sidebar.header("Filters")
    categories = df['Category'].unique().tolist()
    selected = st.sidebar.multiselect("Select Category", categories, default=categories)

    filtered_df = df[df['Category'].isin(selected)]

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    # KPIs
    total, avg, top = calculate_financial_kpis(filtered_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spend", f"${total:,.2f}")
    col2.metric("Monthly Avg", f"${avg:,.2f}")
    col3.metric("Top Category", top)

    # -----------------------------
    # BUDGET VS ACTUAL
    # -----------------------------
    budget = {
        'Housing': 1500,
        'Food & Dining': 800,
        'Transport': 400,
        'Entertainment': 300,
        'Healthcare': 500,
        'Utilities': 300,
        'Shopping': 400
    }

    st.subheader("💸 Budget vs Actual Spending")

    budget_df = filtered_df.groupby('Category')['Amount'].sum().reset_index()
    budget_df['Budget'] = budget_df['Category'].map(budget)

    fig_budget = px.bar(
        budget_df,
        x='Category',
        y=['Amount', 'Budget'],
        barmode='group'
    )
    st.plotly_chart(fig_budget, use_container_width=True)

    # -----------------------------
    # SMART ALERTS
    # -----------------------------
    st.subheader("🚨 Smart Alerts")

    alerts = []
    for _, row in budget_df.iterrows():
        if row['Amount'] > row['Budget']:
            alerts.append(f"⚠️ Overspending in {row['Category']}")

    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("✅ You are within budget!")

    # -----------------------------
    # FINANCIAL HEALTH SCORE
    # -----------------------------
    st.subheader("🧠 Financial Health Score")

    overspend_count = sum(budget_df['Amount'] > budget_df['Budget'])
    score = max(0, 100 - overspend_count * 15)

    if score > 80:
        st.success(f"Excellent! Score: {score}")
    elif score > 50:
        st.warning(f"Moderate Control. Score: {score}")
    else:
        st.error(f"Poor Spending Habits! Score: {score}")

    # -----------------------------
    # CHARTS
    # -----------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Monthly Trend")
        trend = (
            filtered_df
            .set_index('Date')
            .resample('ME')['Amount']
            .sum()
            .reset_index()
        )
        fig1 = px.line(trend, x='Date', y='Amount', markers=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("📊 Category Distribution")
        cat_data = filtered_df.groupby('Category')['Amount'].sum().reset_index()
        fig2 = px.pie(cat_data, values='Amount', names='Category', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # DAILY TREND
    # -----------------------------
    st.subheader("📅 Daily Spending Pattern")

    daily = (
        filtered_df
        .set_index('Date')
        .resample('D')['Amount']
        .sum()
        .reset_index()
    )

    fig_daily = px.line(daily, x='Date', y='Amount')
    st.plotly_chart(fig_daily, use_container_width=True)

    # -----------------------------
    # ANOMALY
    # -----------------------------
    st.markdown("---")
    st.subheader("⚠️ Anomaly Detection")

    anomalies = filtered_df[filtered_df['Anomaly'] == 'Anomalous']
    st.dataframe(anomalies.sort_values(by='Amount', ascending=False))

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    st.sidebar.markdown("---")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", csv, "financial_data.csv")


if __name__ == "__main__":
    run_app()