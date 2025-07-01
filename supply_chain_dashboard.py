import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import product
import random

st.set_page_config(page_title="AI-Powered Resilient Supply Chain", layout="wide")

st.title("🚚 GPT-powered Intelligent Supply Chain Dashboard")
st.markdown("### By: IIM Indore | TechNZ | #AI #SupplyChain #Streamlit #Innovation")

# Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    selected_tab = st.radio("Select Module", [
        "Megatrend Tracker", "Dynkin Diagram", "Markov Model",
        "Game Theory", "Forecast Simulation", "RAG Insights"
    ])

# ---------------- TAB 1: Megatrend Tracker ----------------
if selected_tab == "Megatrend Tracker":
    st.subheader("📈 Megatrend Tracker")
    trend = st.selectbox("Select Trend", ["Demand Fluctuation", "Logistics Disruptions", "Inventory Accumulation"])

    years = list(range(2020, 2026))
    values = np.random.randint(50, 150, size=len(years))
    df = pd.DataFrame({"Year": years, "Index": values})

    fig = px.line(df, x="Year", y="Index", title=f"{trend} Over Time")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: Dynkin Diagram ----------------
elif selected_tab == "Dynkin Diagram":
    st.subheader("🔗 Dynkin Diagram (Supply Chain Network)")
    nodes = ["Supplier", "Manufacturer", "Warehouse", "Retailer", "Customer"]
    edges = [("Supplier", "Manufacturer"), ("Manufacturer", "Warehouse"),
             ("Warehouse", "Retailer"), ("Retailer", "Customer"),
             ("Warehouse", "Customer")]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', arrows=True)
    st.pyplot(fig)

# ---------------- TAB 3: Markov Model ----------------
elif selected_tab == "Markov Model":
    st.subheader("🔄 Markov Chain State Transition")
    states = ["Stable", "Disrupted", "Recovering"]
    transition_matrix = np.array([[0.7, 0.2, 0.1],
                                  [0.3, 0.5, 0.2],
                                  [0.2, 0.3, 0.5]])

    state = st.selectbox("Select Initial State", states)
    steps = st.slider("Number of Steps", 1, 10, 5)

    state_idx = states.index(state)
    current = np.zeros(len(states))
    current[state_idx] = 1.0

    history = [current]
    for _ in range(steps):
        current = np.dot(current, transition_matrix)
        history.append(current)

    df = pd.DataFrame(history, columns=states)
    st.line_chart(df)

# ---------------- TAB 4: Game Theory ----------------
elif selected_tab == "Game Theory":
    st.subheader("🎲 Game Theory Payoff Matrix")
    st.write("Two players: Manufacturer and Retailer")

    M = [[3, 1], [5, 2]]  # Manufacturer Payoff
    R = [[2, 0], [3, 1]]  # Retailer Payoff

    strategy_labels = ["High Inventory", "Low Inventory"]
    st.write("**Manufacturer Payoff Matrix:**")
    st.dataframe(pd.DataFrame(M, index=strategy_labels, columns=strategy_labels))

    st.write("**Retailer Payoff Matrix:**")
    st.dataframe(pd.DataFrame(R, index=strategy_labels, columns=strategy_labels))

    st.write("**Note:** Equilibrium visualized is hypothetical and strategic, not solved.")

# ---------------- TAB 5: Forecast Simulation ----------------
elif selected_tab == "Forecast Simulation":
    st.subheader("📊 Forecast Simulation (No TensorFlow)")
    st.write("Simulated disruption forecast using NumPy")

    time = np.arange(100)
    signal = np.sin(time * 0.1) + np.random.normal(0, 0.1, 100)

    window = 10
    forecast = [np.mean(signal[i:i+window]) + np.random.normal(0, 0.05) for i in range(len(signal) - window)]

    df_forecast = pd.DataFrame({
        "Time": time[window:],
        "Forecast": forecast
    })
    st.line_chart(df_forecast.set_index("Time"))

# ---------------- TAB 6: RAG Insights ----------------
elif selected_tab == "RAG Insights":
    st.subheader("📚 RAG-Powered Dynamic Insight")
    query = st.text_input("Enter a query for insight (simulated)", "How can I reduce ripple effects?")
    if st.button("Generate Insight"):
        response = f"✅ For query '{query}':\nIntegrate real-time visibility tools with predictive buffers at key nodes. Use AI to model propagation of delays from suppliers to customer."
        st.success(response)

st.markdown("---")
st.caption("Created by Group 2, PGDCS-2, IIT Jammu | Powered by Streamlit")