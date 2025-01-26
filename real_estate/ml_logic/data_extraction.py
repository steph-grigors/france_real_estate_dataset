import os
import pandas as pd
import glob
import time
import numpy as np
from real_estate.params import *

# File paths and configuration
npz_file_path = os.path.join(LOCAL_DATA_PATH, "transactions.npz")
chunk_size = 50000  # Adjust chunk size based on memory availability
output_dir = os.path.join(LOCAL_DATA_PATH, "processed_chunks")
output_file = os.path.join(LOCAL_DATA_PATH, "full_transactions_dataset", "transactions_combined.csv")



def process_npz_chunkwise_to_csv(npz_file_path, chunk_size, output_dir):
    arrays = np.load(npz_file_path)
    keys = arrays.files
    total_rows = len(arrays[keys[0]])  # Assuming all keys have the same length

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and save each chunk
    for start in range(0, total_rows, chunk_size):
        chunk = {}

        for key in keys:
            data = arrays[key]

            # Process string data
            if data.dtype == np.uint8:
                decoded_data = [s.decode("utf-8") for s in data.tobytes().split(b"\x00")]
                chunk[key] = decoded_data[start:start + chunk_size]
            else:
                # Process numeric data
                chunk[key] = data[start:start + chunk_size]

        # Create a DataFrame for the chunk
        chunk_df = pd.DataFrame(chunk)

        # Save the chunk to a CSV file
        output_file = os.path.join(output_dir, f"transactions_chunk_{start}_{start + chunk_size}.csv")
        chunk_df.to_csv(output_file, index=False)
        print(f"Processed and saved rows {start} to {start + chunk_size} out of {total_rows} in {output_file}")

    print("All chunks processed and saved successfully!")



def concatenate_csv_files(output_dir, output_file):
    # Get a list of all CSV files in the output directory
    csv_files = glob.glob(os.path.join(output_dir, "transactions_chunk_*.csv"))

    # Sort the files to maintain the order of chunks
    csv_files.sort()

    # Read and concatenate all the CSV files
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the combined DataFrame to a single CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"All files concatenated and saved to {output_file}")


if __name__ == '__main__':
    # process_npz_chunkwise_to_csv(file_path, chunk_size, output_dir)
    concatenate_csv_files(output_dir, output_file)
