import os
import csv
import time
"""
Script to summarize all action made to a file according to their file naming.
Useful to keep track and understand the dataset.

After this step, you can rename all your files with a consistent name.
"""
time1 = time.time()
# Define the path to your dataset folder
dataset_folder = "Original_data/Cropped_MAX-Seg/"

# Define the output CSV file path
output_csv_file = "Original_data/Info/summary_MAX-Seg.csv"

# Define the column names for the CSV file
column_names = ["Patient", "Side", "Segmentation Method", "Cropped Scan","Cropped Seg->Mask","Canine Segmented"]

# Create a list to store the summarized data
summary_data = []

# Iterate over the subfolders in the dataset folder
for case_folder in ["Bilateral", "Left", "Right"]:
    case = case_folder
    case_folder_path = os.path.join(dataset_folder, case_folder)+"/Segmentation"


    # Iterate over the files in the case folder
    for filename in os.listdir(case_folder_path):
        name = filename.split("_")[0]
        segmentation_method = ""
        cropping = ""

        # Extract the segmentation method from the filename
        if "LFOV" in filename:
            segmentation_method = "Large FOV"
        elif "SFOV" in filename:
            segmentation_method = "Small FOV"
        elif "MS" in filename:
            segmentation_method = "Manual Seg"

        # Check if the filename contains "cropped"
        if not "MS" in filename:
            cropping = "cropped"

        crop_seg=''
        # Append the summarized data to the list
        summary_data.append([name, case, segmentation_method, crop_seg,cropping,"yes"])

# Write the summarized data to the CSV file
with open(output_csv_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(column_names)
    writer.writerows(summary_data)

timeEnd = time.time()
print("Dataset was summarized in ",round(timeEnd - time1,7), "seconds")