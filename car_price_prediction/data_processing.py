import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Setup file path
input_path = "data/raw_data/car.csv"
output_dir = "data/processed_data"
output_path = f"{output_dir}/cleaned_car_data.csv"

os.makedirs(output_dir ,exist_ok=True)

#loading the data
print("Loading data...")
car_data = pd.read_csv(input_path)
print(f'Initial size:{car_data.shape}')

#handling the null records
print("----checking the null values...")
missing_values=car_data.isnull().sum().sum()
print(f"Total null values:{missing_values}")
if missing_values > 0:
    car_data.dropna(inplace=True)
    print("Null values dropped.The new shap",car_data.shape)


#droping the duplicate
car_data.drop_duplicates(inplace=True)

print("----unique values...")
print(car_data.nunique())

#  Feature Engineering
print("\n--- Engineering 'Years_of_Service' ---")
current_year = 2024
car_data['Years_of_Service'] = current_year - car_data['Year']

#  Drop Unnecessary Features
print("\n--- Dropping Unnecessary Features ---")
# Dropping Car_Name to prevent overfitting, and Year since we have Years_of_Service
car_data.drop(['Car_Name', 'Year'], axis=1, inplace=True)

#categorical encoding
df_encoded=pd.get_dummies(car_data,drop_first=True)

# 7. Handle Highly Correlated "Related" Features
corr_matrix = df_encoded.corr().abs()
print(corr_matrix)

#creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Final Encoded Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"plot/corr_matrix.png")

# 8. Save the Processed Data
print(f"\n--- Saving final robust data to {output_path} ---")
df_encoded.to_csv(output_path, index=False)
print(f"Final Data Shape: {df_encoded.shape}")
print(" Phase 1: Data processing is  Complete!")
#the first stage is complete
