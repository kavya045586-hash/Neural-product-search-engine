#TO ACTIVATE THE VIRTUAL ENVIRONMENT venv312\Scripts\activate
# import pandas as pd

# # Path to your file
# file_path = r"C:\Users\nagar\Downloads\parquet\electronics_sample_2M (1).parquet"

# # Read parquet file
# df = pd.read_parquet(file_path)

# # Show first 5 rows
# print(df.head())
import pandas as pd

# Path to your parquet file
parquet_file = r"C:\Users\nagar\Downloads\parquet\electronics_sample_2M (1).parquet"

# Read parquet file
df = pd.read_parquet(parquet_file)

# Convert to CSV
csv_file = r"C:\Users\nagar\Downloads\parquet\electronics_sample.csv"
df.to_csv(csv_file, index=False)

print("Conversion complete! CSV saved at:", csv_file)