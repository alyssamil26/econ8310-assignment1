# Import libraries/functions
from pygam import LinearGAM, s, f, l
import pandas as pd
import numpy as np

# Read in data
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

#variables
x = data[['year', 'month', 'day', "hour"]].values
y = data['trips'].values

model = LinearGAM(
    s(0) +
    f(1) +
    f(2) +
    s(3)
)

modelFit = model.gridsearch(x.values, y)

# --- Build Future: Jan of the year AFTER the max training year (744 hours) ---
train_last_year = int(data["year"].max())
future_start = pd.Timestamp(train_last_year + 1, 1, 1, 0, 0, 0)
future_index = pd.date_range(start=future_start, periods=744, freq="H")

future_df = pd.DataFrame({
    "year":  future_index.year,
    "month": future_index.month,
    "day":   future_index.day,
    "hour":  future_index.hour,
})

# Make numeric ints
future_df = future_df.astype({"year": int, "month": int, "day": int, "hour": int})

X_future = future_df[["year", "month", "day", "hour"]].values

# Prediction
pred = modelFit.predict(X_future)
pred = np.asarray(pred).ravel()   # ensure 1D numeric array

