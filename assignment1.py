# Import libraries/functions
from pygam import LinearGAM, s, f, l
import pandas as pd

# Read in data
data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

#variables
x = data[['year', 'month', 'day', "hour"]]
y = data['trips']

model = LinearGAM(
    s(0) +
    f(1) +
    f(2) +
    s(3)
)

modelFit = model.gridsearch(x.values, y)

pred = model.predict(x)