import os
import sys

def merge_data_files(in_dir):
    """Merge all log files in a directory (e.g. from loading incomplete runs)
        to make a single data file
    Note: This works because I have only used .log files for initial runs
        and all slurm files come indexed in chronological order
    Note: Takes advantage of plot.py's robustness (read 'creates files with ugly formatting')"""

    # Collect and order files
    in_files = []
    for f in os.listdir(in_dir):
        f_name = in_dir + '/' + f
        if '.log' in f:
            in_files = [f_name] + in_files
        elif 'slurm' in f:
            # For data files, add it to files
            in_files.append(f_name)

    # Add the contents of the input files to the merged file
    with open(in_dir + "/merged_log_data.txt", 'a') as out_file:
        if in_files == []:
            print("No files found for merging in directory {}".format(in_dir))
        else:
            for f_name in in_files:
                with open(f_name, 'r') as in_file:
                    for line in in_file:
                        out_file.write(line)

                # Delete the input files once they've been merged
                os.remove(f)
                    
if __name__ == "__main__":
    in_dir = sys.argv[1]
    merge_data_files(in_dir)