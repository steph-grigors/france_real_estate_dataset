import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from real_estate.params import *

# File paths and configuration
npz_file_path = os.path.join(LOCAL_DATA_PATH, "raw_dataset", "transactions.npz")
output_file = os.path.join(LOCAL_DATA_PATH, "raw_dataset_full", "transactions.parquet")

def process_npz_to_parquet(npz_file_path, output_file, chunk_size= CHUNK_SIZE):
    arrays = np.load(npz_file_path)
    keys = arrays.files
    total_rows = len(arrays[keys[0]])  # Assuming all keys have the same length


    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create a Parquet writer (appending chunks instead of creating multiple files)
    parquet_writer = None

    for start in range(0, total_rows, chunk_size):
        chunk = {}

        for key in keys:
            data = arrays[key]

            # Handle string decoding if stored as uint8
            if data.dtype == np.uint8:
                decoded_data = [s.decode("utf-8") for s in data.tobytes().split(b"\x00")]
                chunk[key] = decoded_data[start:start + chunk_size]
            else:
                chunk[key] = data[start:start + chunk_size]

        # Convert chunk to a DataFrame
        chunk_df = pd.DataFrame(chunk)

        # Convert DataFrame to Arrow Table for Parquet
        table = pa.Table.from_pandas(chunk_df)

        # Write chunk to Parquet
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_file, table.schema)

        parquet_writer.write_table(table)

        print(f"Processed and saved rows {start} to {start + chunk_size} out of {total_rows}")

    if parquet_writer:
        parquet_writer.close()

    print(f"All data successfully processed and saved to {output_file} as Parquet.")

if __name__ == '__main__':
    process_npz_to_parquet(npz_file_path, output_file, chunk_size=CHUNK_SIZE)
