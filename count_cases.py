import csv
import pandas as pd
import argparse

# Path to the CSV file
parser = argparse.ArgumentParser(description="Count the number of cases with labels")
parser.add_argument("--csv_file", help="Path to the CSV file")
args = parser.parse_args()
csv_file = args.csv_file
# Initialize counters
l_cases = 0
r_cases = 0
both_cases = 0

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    ct=0
    for row in reader:
        ct+=1
        label_r = row['Label_R']
        label_l = row['Label_L']
        #change type of row
        label_r = int(label_r) if label_r else None
        label_l = int(label_l) if label_l else None
        print(f'======row {ct} {row}======')
        print('Name', row['Name']  )
        print('label r', label_r)
        print('label l', label_l)

        # Check if Label_R is None and Label_L has a class
        #use pd.isna
        if (pd.isna(label_r) and label_l is not None) or ( label_r is None and label_l is not None):
            l_cases += 1


        # Check if Label_R has a class and Label_L is None
        elif (label_r is not None and pd.isna(label_l)) or (label_r is not None and label_l is None):
            r_cases += 1


        # Check if both Label_R and Label_L have classes
        elif label_r is not None and label_l is not None:
            both_cases += 1


        else:
            print("This case has no labels")

# Print the counts
print('number lines csv', reader.line_num)
print(f"Number of L cases: {l_cases}")
print(f"Number of R cases: {r_cases}")
print(f"Number of cases with both labels: {both_cases}")
sum_cases = l_cases + r_cases + both_cases
if sum_cases == reader.line_num-1:
    print('[SUCCESS] All cases have been counted')