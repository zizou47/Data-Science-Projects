import pandas as pd
import requests
import json

def send_batch(data, url, headers):
    response = requests.post(url, headers=headers, json={"data": data})
    return response.json()

def main():
    url = 'http://127.0.0.1:8000/predict'  # API endpoint
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}

    # Load your CSV data
    csv_file = r'C:\Users\yazid\Desktop\Bank churn\outputs\test_csv.csv'
    df = pd.read_csv(csv_file)

    # Convert the DataFrame to a list of dictionaries
    data_list = df.to_dict(orient='records')

    # Define batch size
    batch_size = 10
    results = []

    # Process and send data in batches
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        result = send_batch(batch, url, headers)
        results.append(result)
        print(f"Processed batch {i//batch_size + 1}: {result}")

    # Optionally save results to a file
    with open('batch_results.json', 'w') as result_file:
        json.dump(results, result_file, indent=4)

if __name__ == "__main__":
    main()
