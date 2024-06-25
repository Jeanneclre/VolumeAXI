import csv
import argparse
import os
import pandas as pd

def main(args):
    input_file = args.input
    output_file = args.output
    # Create a dictionary to store the combined rows
    combined_rows = {}

    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        # Iterate over each row in the input file
        for row in reader:
            path, name, label = row

            # Extract the patient number from the name
            patient_number = name.split('_')[0]
            side = name.split('_')[1]
            # Check if a row with the same patient number already exists
            if patient_number in combined_rows:
                # Add the label to the corresponding column
                if side == 'R':
                    combined_rows[patient_number]['Label R'] = label

                elif side == 'L':
                    combined_rows[patient_number]['Label L'] = label

            else:
                # Create a new entry for the patient number
                if side == 'R':
                    combined_rows[patient_number] = {
                        'Path': path,
                        'Name': name,
                        'Label R': label,
                        'Label L': 'nan'
                    }
                elif side == 'L':
                    combined_rows[patient_number] = {
                        'Path': path,
                        'Name': name,
                        'Label R': 'nan',
                        'Label L': label
                    }

    # Write the combined rows to the output CSV file
    output_dir = os.path.dirname(input_file)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['Path', 'Name', 'Label R', 'Label L','Label comb'])

        # Write the combined rows
        for row in combined_rows.values():
            if row['Label R'] != 'nan' and row['Label L'] != 'nan':
                if int(row['Label R']) > int(row['Label L']):
                    #concat the labels in the order of the highest label
                    comb_label = row['Label L']+ row['Label R']
                else :
                    comb_label = row['Label R']+ row['Label L']
            elif row['Label R'] != 'nan' and row['Label L'] == 'nan':
                comb_label = row['Label R']
            elif row['Label R'] == 'nan' and row['Label L'] != 'nan':
                comb_label = row['Label L']

            writer.writerow([row['Path'], row['Name'], row['Label R'], row['Label L'],comb_label])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine labels for left and right images')
    parser.add_argument('--input', required=True, type=str, help='Input CSV file')
    parser.add_argument('--output', required=True, type=str, help='Output CSV filename only')
    args = parser.parse_args()
    main(args)