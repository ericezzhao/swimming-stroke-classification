# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:10:31 2024

@author: zhaoez
"""

import os

# folder path
folder_path = r"C:\Users\zhaoez\Desktop\stroke classification business report\strokeanalysis - translated\09 startDive( .from a block)"

# check for if path exists
if not os.path.exists(folder_path):
    print("folder path does not exist.")
    exit()

files = os.listdir(folder_path)
# debugging by printing all files found
print("files in folder:", files)

# renames all files
for file_name in files:
    # debugging each file name
    print(f"processing: {file_name}")
    
    # all files are formatted: 'sclice-<number>.csv'
    if file_name.lower().startswith("sclice-") and file_name.lower().endswith(".csv"):
        try:
            # keep number for the file
            file_number = file_name.split('-')[1].split('.')[0]
            new_name = f"data-{file_number}.csv"
            
            # full paths
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(folder_path, new_name)
            
            os.rename(old_file_path, new_file_path)

            print(f"renamed: {file_name} -> {new_name}")
        # error
        except Exception as e:
            print(f"rrror processing {file_name}: {e}")
    # catch
    else:
        print(f"Skipping: {file_name}")

