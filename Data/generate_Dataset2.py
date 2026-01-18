import pandas as pd
import numpy as np

rows = 20000

df = pd.DataFrame({
    "Patient_Number": np.random.randint(1, 2001, rows),
    "Day_Number": np.random.randint(1, 11, rows),  # last 10 days
    "Physical_activity": np.random.randint(1000, 15000, rows)
})

df.to_csv("dataset2.csv", index=False)
print("dataset2.csv created successfully!")
