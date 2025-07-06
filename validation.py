import streamlit as st
import pandas as pd
import time
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
st.title("📡 WSN Fault Detection & Validation")

# File Upload - Training Data
st.subheader("📂 Upload Training Data")
uploaded_train = st.file_uploader("Upload your sensor data CSV for training", type=["csv"])

if uploaded_train is not None:
    # Read Training Data
    df_train = pd.read_csv(uploaded_train, parse_dates=["Timestamp"])
    st.write("### Preview of Training Data")
    st.write(df_train.head())

    # Data Preprocessing
    features = ["SensorData", "BatteryLife", "Temperature"]
    target = "IsFaulty"

    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(df_train[features], df_train[target], test_size=0.2, random_state=42)

    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

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

    for name, model in models.items():
        st.write(f"🔄 Training {name} model...")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        start_time = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        end_time = time.time()
        st.success(f"✅ {name} Model Training Complete! (Time: {end_time - start_time:.2f} sec)")

    # Test Data Upload
    st.subheader("📂 Upload Test Data (For Validation)")
    uploaded_test = st.file_uploader("Upload another CSV file for testing", type=["csv"])

    if uploaded_test is not None:
        df_test = pd.read_csv(uploaded_test, parse_dates=["Timestamp"])
        st.write("### Preview of Test Data")
        st.write(df_test.head())

        # Prepare Test Data (Without `IsFaulty` for Prediction)
        X_test_actual = df_test[features]  # Preserve actual test data
        X_test = scaler.transform(X_test_actual)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Store Predictions & Measure Time
        y_actual = df_test["IsFaulty"].values  # Actual labels from test file
        y_preds = {}
        total_prediction_times = {}
        per_sample_times = {}

        for name, model in models.items():
            start_time = time.time()
            y_preds[name] = (model.predict(X_test).flatten() > 0.5).astype(int)
            end_time = time.time()
            
            total_prediction_times[name] = end_time - start_time  # Total prediction time
            per_sample_times[name] = total_prediction_times[name] / len(df_test)  # Avg time per sample

        # Display Results
        st.subheader("📊 Model Performance on Test Data")

        results = []
        for name in models.keys():
            acc = accuracy_score(y_actual, y_preds[name])
            prec = precision_score(y_actual, y_preds[name])
            rec = recall_score(y_actual, y_preds[name])
            f1 = f1_score(y_actual, y_preds[name])
            results.append([name, acc, prec, rec, f1, total_prediction_times[name], per_sample_times[name]])

        # Display Results in a Table
        results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "Total Prediction Time (s)", "Prediction Time per Sample (s)"])
        st.write(results_df)

        # Find Best Model
        best_model = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
        st.success(f"🏆 Best Model: **{best_model}** with Accuracy: {results_df.loc[results_df['Model'] == best_model, 'Accuracy'].values[0]:.2f}")

        # Confusion Matrices
        st.subheader("📊 Confusion Matrices")
        for name in models.keys():
            cm = confusion_matrix(y_actual, y_preds[name])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "Faulty"], yticklabels=["Normal", "Faulty"], ax=ax)
            ax.set_title(f"{name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # Line Graph - Actual vs. Predicted Faults
        st.subheader("📈 Actual vs. Predicted Faults")
        plt.figure(figsize=(8, 4))
        colors = {"CNN": "blue", "RNN": "green", "LSTM": "red"}

        actual_counts = pd.Series(y_actual).value_counts().sort_index()
        plt.plot(["Normal", "Faulty"], actual_counts, marker="o", linestyle="--", color="black", label="Actual", linewidth=2)

        for name in models.keys():
            pred_counts = pd.Series(y_preds[name]).value_counts().sort_index()
            plt.plot(["Normal", "Faulty"], pred_counts, marker="o", linestyle="-", color=colors[name], label=f"Predicted ({name})", linewidth=2)

        plt.xlabel("Fault Status")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Show Individual Predictions
        st.subheader("📜 Individual Predictions")
        predictions_df = df_test[["Timestamp"]].copy()
        predictions_df["Actual IsFaulty"] = y_actual
        for name in models.keys():
            predictions_df[f"Predicted ({name})"] = y_preds[name]
        st.write(predictions_df)

        st.success("✅ Validation Complete!")
