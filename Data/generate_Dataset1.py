import pandas as pd
import numpy as np

n = 2000

df = pd.DataFrame({
    "Patient_Number": np.arange(1, n+1),
    "Blood_Pressure_Abnormality": np.random.choice([0,1], n),
    "Level_of_Hemoglobin": np.round(np.random.uniform(9.0, 16.0, n), 2),
    "Genetic_Pedigree_Coefficient": np.round(np.random.uniform(0.0, 1.0, n), 3),
    "Age": np.random.randint(18, 80, n),
    "BMI": np.round(np.random.uniform(18.0, 40.0, n), 1),
    "Sex": np.random.choice([0,1], n),
    "Pregnancy": np.random.choice([0,1], n),
    "Smoking": np.random.choice([0,1], n),
    "salt_content_in_the_diet": np.random.randint(1500, 4000, n),
    "alcohol_consumption_per_day": np.random.randint(0, 60, n),
    "Level_of_Stress": np.random.choice([1,2,3], n),
    "Chronic_kidney_disease": np.random.choice([0,1], n),
    "Adrenal_and_thyroid_disorders": np.random.choice([0,1], n),
})

df.to_csv("dataset1.csv", index=False)
print("dataset1.csv created successfully!")
