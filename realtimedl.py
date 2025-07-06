import streamlit as st
import pandas as pd
import time
import base64
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("📡 Real-time WSN Fault Detection")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your sensor data CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV with proper format
    df = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])
    st.subheader("🔍 Preview of Uploaded Data")
    st.write(df.head())

    # Data Preprocessing
    features = ["SensorData", "BatteryLife", "Temperature"]
    target = "IsFaulty"

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define Models
    def create_lstm():
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1, activation='sigmoid')
        ])
        return model

    def create_cnn():
        model = Sequential([
            Conv1D(32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        return model

    def create_rnn():
        model = Sequential([
            SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            SimpleRNN(50),
            Dense(1, activation='sigmoid')
        ])
        return model

    # Initialize and Train Models
    models = {
        "CNN": create_cnn(),
        "RNN": create_rnn(),
        "LSTM": create_lstm()
    }
    def play_alert():
        sound_file = r"C:\Users\nidhi\OneDrive\Documents\MITInternship\code\alert.mp3"  # Ensure you have an alert sound file
        with open(sound_file, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
            st.markdown(md, unsafe_allow_html=True)

    for name, model in models.items():
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        st.success(f"{name} Model Training Complete!")

    # Layout for Displaying Models Side-by-Side
    col1, col2, col3 = st.columns(3)

    # Streamlit placeholders for graphs
    dist_placeholder = st.empty()
    heatmap_placeholder = st.empty()
    temp_impact_placeholder = st.empty()
    battery_impact_placeholder = st.empty()

    # **📊 FEATURE DISTRIBUTION (REAL-TIME UPDATING)**
    with dist_placeholder.container():
        st.subheader("📊 Feature Distribution (Normal vs. Faulty)")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, feature in enumerate(features):
            sns.histplot(df, x=feature, hue="IsFaulty", kde=True, ax=axes[idx], palette=["green", "red"])
            axes[idx].set_title(f"{feature} Distribution")
        st.pyplot(fig)

    # **📈 CORRELATION HEATMAP**
    with heatmap_placeholder.container():
        st.subheader("📈 Correlation Heatmap")
        corr_matrix = df[features + [target]].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # **🌡️ TEMPERATURE IMPACT ON FAULTS**
    with temp_impact_placeholder.container():
        st.subheader("🌡️ Fault Rate by Temperature")
        df["Temp_Bin"] = pd.cut(df["Temperature"], bins=[0, 10, 20, 30, 40, 50], labels=["0-10", "10-20", "20-30", "30-40", "40-50"])
        temp_fault_rate = df.groupby("Temp_Bin")["IsFaulty"].mean()
        fig, ax = plt.subplots()
        sns.barplot(x=temp_fault_rate.index, y=temp_fault_rate.values, ax=ax, palette="Reds")
        st.pyplot(fig)

    # **🔋 BATTERY IMPACT ON FAULTS**
    with battery_impact_placeholder.container():
        st.subheader("🔋 Fault Rate by Battery Level")
        df["Battery_Bin"] = pd.cut(df["BatteryLife"], bins=[0, 20, 40, 60, 80, 100], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
        battery_fault_rate = df.groupby("Battery_Bin")["IsFaulty"].mean()
        fig, ax = plt.subplots()
        sns.barplot(x=battery_fault_rate.index, y=battery_fault_rate.values, ax=ax, palette="Reds")
        st.pyplot(fig)

    st.success("✅ Graphs Updated Successfully!")
# Real-time Sensor Data Streaming & Fault Detection
    st.subheader("Real-time Sensor Data Streaming & Fault Detection")

    fault_prob_placeholders = {
        "CNN": st.empty(),
        "RNN": st.empty(),
        "LSTM": st.empty()
    }

    metrics_placeholder = st.empty()
    best_model_placeholder = st.empty()

    # Data Storage for Live Charts
    metrics_data = {name: {"accuracy": [], "precision": [], "recall": [], "f1": []} for name in models.keys()}
    chart_placeholders = {
        "CNN": st.line_chart(pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])),
        "RNN": st.line_chart(pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])),
        "LSTM": st.line_chart(pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"]))
    }

    # Placeholders for Confusion Matrices
    confusion_placeholders = {name: st.empty() for name in models.keys()}

    y_preds = {}

    for name, model in models.items():
        y_preds[name] = (model.predict(X_test).flatten() > 0.5).astype(int)

    model_metrics = {}

    for name in models.keys():
        acc = accuracy_score(y_test, y_preds[name])
        prec = precision_score(y_test, y_preds[name])
        rec = recall_score(y_test, y_preds[name])
        f1 = f1_score(y_test, y_preds[name])
        model_metrics[name] = (acc + prec + rec + f1) / 4  # Aggregate Score

    best_model = max(model_metrics, key=model_metrics.get)

    for index, row in df.iterrows():
        input_data = np.array([[row["SensorData"], row["BatteryLife"], row["Temperature"]]])
        input_data = scaler.transform(input_data)
        input_data = np.reshape(input_data, (1, input_data.shape[1], 1))

        fault_probs = {name: model.predict(input_data)[0][0] for name, model in models.items()}

        for name, placeholder in fault_prob_placeholders.items():
            fault_status = "FAULTY" if fault_probs[name] > 0.5 else "NORMAL"
            placeholder.write(f"**{name} Prediction**\nFault Probability: {fault_probs[name]:.2f} ({fault_status}) - Actual: {row['IsFaulty']}")

            if fault_status == "FAULTY":
                play_alert()  # Trigger sound alert
        # Compute Metrics Dynamically
        for name, model in models.items():
            pred_class = (fault_probs[name] > 0.5).astype(int)

            acc = accuracy_score(y_test[:index+1], y_preds[name][:index+1]) if index > 10 else 0
            prec = precision_score(y_test[:index+1], y_preds[name][:index+1], zero_division=0) if index > 10 else 0
            rec = recall_score(y_test[:index+1], y_preds[name][:index+1], zero_division=0) if index > 10 else 0
            f1 = f1_score(y_test[:index+1], y_preds[name][:index+1], zero_division=0) if index > 10 else 0

            # Store data for plotting
            metrics_data[name]["accuracy"].append(acc)
            metrics_data[name]["precision"].append(prec)
            metrics_data[name]["recall"].append(rec)
            metrics_data[name]["f1"].append(f1)

            # Update charts dynamically
            with chart_placeholders[name].container():
                st.line_chart(pd.DataFrame({
                    "Accuracy": metrics_data[name]["accuracy"],
                    "Precision": metrics_data[name]["precision"],
                    "Recall": metrics_data[name]["recall"],
                    "F1": metrics_data[name]["f1"]
                }))
                st.markdown(f"**Model: {name}**", unsafe_allow_html=True)

        # Update Confusion Matrices in Real-Time
        for name, placeholder in confusion_placeholders.items():
            cm = confusion_matrix(y_test[:index+1], y_preds[name][:index+1]) if index > 10 else np.array([[0, 0], [0, 0]])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "Faulty"], yticklabels=["Normal", "Faulty"], ax=ax)
            ax.set_title(f"{name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            placeholder.pyplot(fig)