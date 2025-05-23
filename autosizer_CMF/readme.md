Made some edits to the pipeline. This works on unedited images, though you usually have to do some additional cropping or blacking out artifacts between steps 1 and 2. 

Create a folder for analysis 
Edit Python files with proper paths:
Step 1: Change the input folder for the given analysis. The checkpoint path and pytesseract path need to be defined once. If necessary, you can change other input parameters specified in the file. 
Labeling GUI: No edits required to run, just check dependencies 
Step 2: Change path names, input_folder, red_dot_csv_folder, output_folder, output_csv, debug_folder, warning_file
Step 3: Change path names, subset_folder, results_csv, tag_ids_file, final_output_folder, output_file, warning_file, calib_folder, and orientation. Depending on the TagID file, check that all columns are defined in keep_cols at the end of the script. 
