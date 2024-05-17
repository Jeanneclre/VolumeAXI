import os
import csv
import argparse
import pandas as pd

def convert_xlsx_to_csv(filename,out_dir,augm):
    '''
    Function to convert xlsx to csv
    input: filename
    output: csv file
    '''


    outputfilename = os.path.basename(filename).split('.')[0] + ".csv"
    outname = os.path.join(out_dir, outputfilename)
    read_file = pd.read_excel(filename)
    # convert xlsx to csv
    read_file.to_csv(outname,
                    index = None,
                    header=True)

    df = pd.DataFrame(pd.read_csv(outname))

    if augm:
        # copy and paste in new rows each row of the filename
        df= pd.concat([df,df],ignore_index=True)

    return df, outname


def create_csv_input(input_folder, output_file, words_list=['_L','_R','_MB', '_ML', '_MR'], side='Left'):
    side_trad ={
        'Left':'_L',
        'Right':'_R'
    }
    # Get the list of files in the input folder
    files = os.listdir(input_folder)

    # Initialize the CSV data
    csv_data = [['Path','Name']]
    idx = 0
    # Iterate over the files
    for file_name in files:
        idx += 1
        # Get the base name of the file
        base_name = os.path.basename(file_name)

        patient_name = None
        for word in words_list:
            if str(word) in base_name:
                # Extract the patient name and append it to the CSV data
                patient_name = base_name.split('_')[0] + word
                break
        if patient_name is None:
            patient_name = base_name.split('_')[0] + side_trad[side]

        # Get the full path of the file
        file_path = os.path.join(input_folder, file_name)

        csv_data.append([file_path,patient_name])

    # Write the CSV data to the output file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"Created CSV file with {idx} patients: {output_file}")

def add_info(input_csv_path,label_csv,patient_column,label_column,words_list,side='Left'):
    '''
    Add the labels of the initial file to the new csv file

    '''
    #check the extension of the file
    ext = os.path.splitext(label_csv)[1]
    if ext == '.xlsx':
        df_label, outname = convert_xlsx_to_csv(label_csv,os.path.dirname(label_csv),augm=False)
    else:
        df_label = pd.read_csv(label_csv)

    # Read the input_csv
    input_csv = pd.read_csv(input_csv_path)
    # Read the patient of label_csv and add the label to the good patient in input_csv
    for index,row in input_csv.iterrows():
        # jump first row because it's the header
        # if index == 0:
        #     continue
        patient = row['Name']

        # cpt=0
        for word in words_list:
            if word in patient:
                if word == '_ML':
                    patient = patient.replace('ML','R')
                if word == '_MB' and side == 'Left':
                    patient = patient.replace('MB','R')

                if word == '_MB' and side == 'Right':
                    patient = patient.replace('MB','L')
                if word == '_MR':
                    patient = patient.replace('MR','L')

        # Check if the patient is in the label_csv
        if f'{patient}' in df_label[patient_column].values:
            # Add the label to the input_csv
            label = df_label.loc[df_label[patient_column] == patient, label_column].values[0]
            # add a label column to the input csv if it does not exist already
            if 'Label' not in input_csv.columns:
                input_csv['Label'] = None

            input_csv.loc[index,'Label'] = label

        #save the new csv
        input_csv.to_csv(input_csv_path, index=False)






if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Create CSV input")
    parser.add_argument("--input_folder", help="Path to the input folder with the scans")
    parser.add_argument("--output", help="Path to the output CSV file")
    parser.add_argument("--words_list", nargs='+', help="List of words to match in the patient name", default=['_L','_R','_MB', '_ML', '_MR'])


    parser.add_argument("--label_file", help="Path to the input file with the labels (can be xlsx or csv)")
    parser.add_argument("--patient_column", type=str, help="Column name of the patient in the input label file")
    parser.add_argument("--label_column", type=str, help="Column name of the label in the input label file")
    parser.add_argument("--side", type=str, help="Side of the canine you're working on", default='Left')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(args.output)
    create_csv_input(args.input_folder, args.output)
    add_info(args.output,args.label_file,args.patient_column,args.label_column,args.words_list,args.side)
