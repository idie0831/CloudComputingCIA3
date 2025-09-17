import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn

# ---------------------------------
# 1. Load Dataset
# ---------------------------------
st.title("Air Quality Forecast using LSTM 🚀")

@st.cache_data
def load_data():
    df = pd.read_csv(
        "AirQualityUCI.csv",
        sep=";",           
        decimal=",",       
        encoding="latin1", 
        low_memory=False
    )
    df = df.dropna(axis=1, how='all')   # drop completely empty cols
    df = df.dropna(how="all")           # drop empty rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df

df = load_data()
st.subheader("Dataset Preview")
st.write("Data shape:", df.shape)
st.write(df.head())

# ---------------------------------
# 2. User Inputs
# ---------------------------------
st.sidebar.header("🔧 Model Settings")
target_col = st.sidebar.selectbox("Select pollutant to predict:", df.columns[2:10])  # choose target
seq_length = st.sidebar.slider("Sequence length (hours):", 12, 72, 24, step=6)
hidden_size = st.sidebar.slider("Hidden layer size:", 10, 200, 50, step=10)
epochs = st.sidebar.slider("Training epochs:", 1, 20, 5)

# ---------------------------------
# 3. EDA
# ---------------------------------
st.subheader("Exploratory Data Analysis")

# Missing values heatmap
st.write("Missing Values Heatmap")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------------------------------
# 4. Preprocessing
# ---------------------------------
st.subheader(f"Preprocessing for target: {target_col}")

df_target = df[[target_col]].copy()
df_target = df_target.interpolate(method="linear")

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_target)

st.line_chart(df_scaled)

# ---------------------------------
# 5. Sequence Prep
# ---------------------------------
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

X, y = create_sequences(df_scaled, seq_length)
X_train, y_train = X[:int(0.8*len(X))], y[:int(0.8*len(y))]
X_test, y_test = X[int(0.8*len(X)):], y[int(0.8*len(y)):]

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# ---------------------------------
# 6. LSTM Model
# ---------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq),1,-1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

st.subheader("Model Training")
model = LSTMModel(hidden_layer_size=hidden_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

progress = st.progress(0)
losses = []
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        loss = loss_function(y_pred, labels)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    progress.progress((i+1)/epochs)

st.line_chart(losses)

# ---------------------------------
# 7. Prediction & Evaluation
# ---------------------------------
with torch.no_grad():
    preds = []
    for seq in X_test:
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        preds.append(model(seq).item())

preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = np.sqrt(mean_squared_error(y_test_inv, preds))
mae = mean_absolute_error(y_test_inv, preds)

st.subheader("Evaluation Metrics")
st.write(f"RMSE: {rmse:.3f}")
st.write(f"MAE: {mae:.3f}")

# ---------------------------------
# 8. Visualization
# ---------------------------------
st.subheader("Actual vs Predicted")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(y_test_inv, label="Actual")
ax.plot(preds, label="Predicted")
ax.legend()
st.pyplot(fig)

# ---------------------------------
# 9. Error Distribution
# ---------------------------------
st.subheader("Prediction Error Distribution")
errors = y_test_inv.flatten() - preds.flatten()
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(errors, bins=30, kde=True, ax=ax)
ax.set_title("Distribution of Prediction Errors")
ax.set_xlabel("Error (Actual - Predicted)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# ---------------------------------
# 10. Manual Prediction
# ---------------------------------
st.sidebar.subheader("🔮 Manual Prediction")
manual_input = st.sidebar.text_area(
    f"Enter last {seq_length} values (comma-separated):",
    value=",".join([str(round(float(v),2)) for v in df_target[target_col].dropna().tail(seq_length).values])
)

if st.sidebar.button("Predict Next Value"):
    try:
        values = [float(x) for x in manual_input.split(",")]
        if len(values) != seq_length:
            st.error(f"❌ Please enter exactly {seq_length} values.")
        else:
            scaled_input = scaler.transform(np.array(values).reshape(-1,1))
            seq = torch.from_numpy(scaled_input).float()
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                    torch.zeros(1, 1, model.hidden_layer_size))
                prediction = model(seq).item()
            prediction = scaler.inverse_transform(np.array([[prediction]]))[0][0]
            st.sidebar.success(f"Predicted Next Value for {target_col}: {prediction:.3f}")
    except:
        st.error("⚠️ Invalid input. Please enter numbers only.")

