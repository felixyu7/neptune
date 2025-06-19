# This script requires the following packages:
# pip install numpy awkward pyarrow
import os
import argparse
import numpy as np
import awkward as ak

def process_file(input_path, output_path, time_variance, dropout_fraction):
    """
    Reads a parquet file, adds noise to the photon data using vectorized
    operations, and saves it to a new file.

    Args:
        input_path (str): Path to the input parquet file.
        output_path (str): Path where the output parquet file will be saved.
        time_variance (float): Variance for the Gaussian timing noise.
        dropout_fraction (float): Fraction of photons to drop.
    """
    try:
        # Read the entire parquet file into an Awkward Array.
        events = ak.from_parquet(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return

    photons = events.photons

    # Ensure there are photons to process
    if len(photons) == 0 or ak.sum(ak.num(photons.t)) == 0:
        # If no photons, just write the original data out
        # and add an empty noise_photons field for schema consistency.
        events["noise_photons"] = photons
        ak.to_parquet(events, output_path)
        return

    # 1. Add random timing noise (vectorized)
    # Generate Gaussian noise with the same structure as the time array.
    time_noise = ak.Array(np.random.normal(0, np.sqrt(time_variance), size=len(ak.flatten(photons.t))))
    # Restore the jagged structure of the noise array to match the photon times.
    time_noise = ak.unflatten(time_noise, ak.num(photons.t))
    noisy_time = photons.t + time_noise

    # 2. Apply random photon efficiency noise (dropout)
    # Create a boolean mask to randomly drop photons.
    dropout_mask = ak.Array(np.random.random(len(ak.flatten(photons.t))) > dropout_fraction)
    # Restore the jagged structure of the mask.
    dropout_mask = ak.unflatten(dropout_mask, ak.num(photons.t))

    # 3. Use ak.zip to create the new 'noise_photons' record.
    # This is more efficient than building it record by record.
    noise_photons = ak.zip({
        "sensor_pos_x": photons.sensor_pos_x[dropout_mask],
        "sensor_pos_y": photons.sensor_pos_y[dropout_mask],
        "sensor_pos_z": photons.sensor_pos_z[dropout_mask],
        "string_id": photons.string_id[dropout_mask],
        "sensor_id": photons.sensor_id[dropout_mask],
        "t": noisy_time[dropout_mask],
        "id_idx": photons.id_idx[dropout_mask],
    }, with_name="photons")

    # Add the newly created noisy data as a new field in the events array.
    events["noise_photons"] = noise_photons
    
    # Save the modified array to a new parquet file.
    ak.to_parquet(events, output_path)

def main():
    """
    Main function to parse command-line arguments and process all parquet files
    in a directory.
    """
    parser = argparse.ArgumentParser(description="Add timing and efficiency noise to photon data in parquet files.")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing the input parquet files.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory where the output parquet files will be saved.")
    parser.add_argument("--time_variance", type=float, default=1.0, 
                        help="Variance for the Gaussian timing noise. Default is 1.0.")
    parser.add_argument("--dropout_fraction", type=float, default=0.1, 
                        help="Fraction of photons to drop for efficiency simulation. Default is 0.1 (10%%).")
    
    args = parser.parse_args()

    # Create the output directory if it doesn't exist.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process each file in the input directory that ends with .parquet.
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".parquet"):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            print(f"Processing {input_path} -> {output_path}")
            process_file(input_path, output_path, args.time_variance, args.dropout_fraction)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()