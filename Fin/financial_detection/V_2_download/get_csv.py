import pandas as pd
import numpy as np

# Load the numpy file
file_path = 'Fin\\financial_detection\\V_2_download\\csv_npy\\preds.npy'
data = np.load(file_path)
y_ = 'Fin\\financial_detection\\V_2_download\\csv_npy\\y.npy'
y = np.load(y_)


# Load the CSV file
csv_file_path = 'Fin\\financial_detection\\V_2_download\\csv_npy\\sample.csv'
csv_data = pd.read_csv(csv_file_path)

csv_data.shape, csv_data.head()

# Extracting the second column from the numpy array
predictions = data[:, 1]

# Creating a new DataFrame for submission
submission = pd.DataFrame({
    # 'index': csv_data['index'],
    'predict': predictions, 
    'y': y
})

# Save to a new CSV file
submission_file_path = 'Fin\\financial_detection\\V_2_download\\csv_npy\\submission.csv'
submission.to_csv(submission_file_path, index=False)

print(submission.head())
