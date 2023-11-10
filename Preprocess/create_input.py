import os
import pandas as pd
import glob
import pathlib
import SimpleITK as sitk
import argparse

####################################################
##                                                ##
##      Create csv file Input with the data       ##
##                                                ##
##                                                ##
####################################################

# Take the xlsx files with all the data/ classification
# Convert it into a csv file without the header and then add the path 
# of the file corresponding to the patient name 

def convert_xlsx_to_csv(filename,out_dir):
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

    return df, outname

def find_path(directory_path:str,side_list:list):
    '''
    Function to find the path of all the files with the side_list in the directory
    input: directory_path (str), side_list (list)
    output: list of the path of the files
    '''
    arguments=[]
    
    for ext in side_list:
        if type(ext) == list:
            arguments.extend(ext)
            
        else:
            arguments.append(ext)
      
    result = {}  # Initialize an empty dictionary
    
    for key in arguments:

        files_matching_key = [] # empty list 'files_matching_key' to store the file paths that end with the current 'key'

        if os.path.isdir(directory_path):
            # Use 'glob.iglob' to find all file paths ending with the current 'key' in the 'path' directory
            # and store the generator object returned by 'glob.iglob' in a variable 'files_generator'
            
            files_list = glob.iglob(os.path.join(directory_path,'**', '*'),recursive=True)
            for i in files_list:
                # check if the k is in the file path
                # and if the file extension is .nii.gz
                if key in i and i.endswith('.nii.gz') :
                    # If the file path ends with the current 'key', append it to the 'files_matching_key' list
                    files_matching_key.append(i)
        
        # Assign the resulting list to the 'key' in the 'result' dictionary
        result[key] = files_matching_key

    # Double the paths list corresponding to the key 'Bilateral'
    # Because in Sara's data file we have 2 rows for each patient with the same 'Bilateral' scans
    specific_key = "Bilateral"

    # Check if the specific key exists in the dictionary
    if specific_key in result:
        # Double the paths list and update the dictionary
        result[specific_key] = result[specific_key] * 2

    # # Convert back to the format you want
    # for key, paths in result.items():
    #     if len(paths) == 1:
    #         result[key] = paths[0]
    #     else:
    #         result[key] = ','.join(paths)
 
    return result
    
def add_path(filename:str,dataframe, path_dict:dict, scans_side:list):
    '''
    Function to add the path of the file corresponding to the patient name
    to the csv file
    '''
    # Add a column 'Path0-3' for better resolution scans
    resolution = 'sp1'
    
    # Extract patient names from the paths and create a mapping from patient name to path
    patient_to_path = {}
    patient_to_path3 = {}
    idx =0
    for side, paths in path_dict.items():
        for path in paths:
               
                if resolution in os.path.basename(path):
                    # Assuming the filename structure as "Clinician_patientName_scan_orientation.nii.gz"
                    if side in path:
                        if "IC" not in path:
                            patientName = pathlib.Path(path).stem.split('_')[1]
                        else:
                            patientName = pathlib.Path(path).stem.split('_')[2]
                        Key = (patientName,side)
                        patient_to_path[patientName,side] = path
                    
                    for index, row in dataframe.iterrows():
                        # convert row['Side'] to a string
                        row['Side'] = str(row['Side'])
                        if side in row['Side'] and patientName in row['Patient'] :
                            dataframe.loc[index,'Path'] = path
                            
                        else :
                            continue
                   
                else: 
                    # Assuming the filename structure as "Clinician_patientName_scan_orientation.nii.gz"
                    if side in path:
                        if "IC" not in path:
                            patientName = pathlib.Path(path).stem.split('_')[1]
                        else:
                            patientName = pathlib.Path(path).stem.split('_')[2]
                        Key3 = (patientName,side)
                        patient_to_path3[patientName,side] = path

                    for index, row in dataframe.iterrows():
                        # convert row['Side'] to a string
                        row['Side'] = str(row['Side'])
                        if side in row['Side'] and patientName in row['Patient'] :
                            dataframe.loc[index,'Path3'] = path
                        else :
                            continue
        idx+=1
        if idx%10==0:
            print(f"Processed {idx} out of {len(path_dict)}")
    # Move the 'Path' column to the front of the dataframe
    cols3 = ["Path3"] + [col for col in dataframe if col != "Path3"]
    dataframe = dataframe[cols3]

    cols = ["Path"] + [col for col in dataframe if col != "Path"]
    dataframe = dataframe[cols]
    # Save the modified CSV
    dataframe.to_csv(filename, index=False)

    print(dataframe)
    return dataframe

def get_data_image(filename:str,df):
    '''
    Function to get the data from the scans (spacing, origin, size, etc)
    and add these informations to the csv file
    '''
    #read the csv file
    #for each row, take the path of the scan, calcul the spacing, origin, size 
    # add column to the csv file with these informations
    # df = pd.read_csv(filename)
    for index, row in df.iterrows():
        path = row['Path3']
        #verify if the path is not empty 
        if type(path) is not str:
            #add column to the csv file
            df.loc[index,'Size_x'] = 'Nan'
            df.loc[index,'Size_y'] = 'Nan'
            df.loc[index,'Size_z'] = 'Nan'

            df.loc[index,'Origin'] = 'Nan'
            df.loc[index,'Spacing_x'] = 'Nan'
            df.loc[index,'Spacing_y'] = 'Nan'
            df.loc[index,'Spacing_z'] = 'Nan'
            
            df.to_csv(filename, index=False)
            
        else:
            #read the image
            img = sitk.ReadImage(path)
            #get the data
            size = img.GetSize()
            origin = img.GetOrigin()
            spacing = img.GetSpacing()
            direction = img.GetDirection()

            #convert data to string
            size_x = str(size[0])
            size_y = str(size[1])
            size_z = str(size[2])

            origin = str(origin)

            spacing_x = str(spacing[0])
            spacing_y = str(spacing[1])
            spacing_z = str(spacing[2])

            direction = str(direction)

            #add column to the csv file
            df.loc[index,'Size_x 0-3'] = size_x
            df.loc[index,'Size_y 0-3'] = size_y
            df.loc[index,'Size_z 0-3'] = size_z

            df.loc[index,'Origin 0-3'] = origin
            
            df.loc[index,'Spacing_x 0-3'] = spacing_x
            df.loc[index,'Spacing_y 0-3'] = spacing_y
            df.loc[index,'Spacing_z 0-3'] = spacing_z
            
            df.to_csv(filename, index=False)

    df.to_csv(filename, index=False)  

def main_input(args):
    directory = args.data_dir
    filename = args.filename
    side_of_scans = args.side
    out_directory = args.out_dir


    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # if the file is a xlsx file, convert it to csv
    if filename.endswith('.xlsx'):
        df, outputFilename = convert_xlsx_to_csv(filename,out_directory)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
        outputFilename = filename
    else:
        print("The file is not a xlsx or csv file")
        return
    path_dict = find_path(directory,side_of_scans)
    df2= add_path(outputFilename,df,path_dict, side_of_scans)
    get_data_image(outputFilename,df2)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='directory with the input files',type=str,default='./')
    parser.add_argument('--filename',help='name of the xlsx file with the data',type=str,default='./')
    parser.add_argument('--out_dir',help='directory where the output files are stored',type=str,default='./output')
    parser.add_argument('--side',help='list with all the side of the scan possible',type=int,default=['Left','Right','Bilateral'])
    args = parser.parse_args() 
    main_input(args)

    # directory = "./data2"
    # filename = "./data2/Sara_Quantification.xlsx"
    # side_of_scans = ["Left","Right","Bilateral"]
    # out_directory = './output'


