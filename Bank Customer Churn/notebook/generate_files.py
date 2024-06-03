import pandas as pd
import random


def split_csv(dataframe, output_folder, num_files, error_percentage):
    # Calculate the chunk size based on the number of files
    chunk_size = len(dataframe) // num_files
    if len(dataframe) % num_files:
        chunk_size += 1

    # Split and save the chunks, adding errors based on the error percentage
    for i in range(num_files):
        chunk = dataframe[i * chunk_size:(i + 1) * chunk_size]

        # Apply errors to a certain percentage of rows
        if error_percentage > 0:
            num_rows_with_error = int(error_percentage / 100 * len(chunk))
            rows_with_error = random.sample(range(len(chunk)), num_rows_with_error)
            
            for row in rows_with_error:
                random_col = random.randint(0, chunk.shape[1] - 1)
                # Select a random numerical error value
                error_value = random.choice([-9, -10000, 0, 9999, float('nan')])
                chunk.iloc[row, random_col] = error_value

        # Explicitly enforce data types for columns (like diagonal)
        if 'diagonal' in chunk.columns:
            chunk['diagonal'] = chunk['diagonal'].astype('float64')

        chunk.to_csv(f"{output_folder}/sample_file_{i+1}.csv", index=False)


# Example usage
df = pd.read_csv("./data/fake_bills.csv")
df_with_only_features = df.drop(["is_genuine", "height_left"], axis=1)

# Adjust the num_files and error_percentage parameters as needed
split_csv(df_with_only_features, "./data/raw_data", num_files=150, error_percentage=30)