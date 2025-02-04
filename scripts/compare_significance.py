import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

import sys
import os

from plot import *

method_list = ['CommNet', 'IC3Net', 'TarMAC', 'Tar-CommNet', 'TarMAC+CAVE', 'GA-Comm', 'GA-Comm+CAVE', 'MAGIC', 'MAGIC+CAVE', 'Dec-TarMAC'] #, 'MAGIC w/o the Scheduler'

def parse_data(files, incl_list, term='Reward'): #, start=0, end=0):
    """
    Parse data from experiment output files for tatistical testing
    Parsing code is copied from plot.parse_plot()
    
    Arguments:
        files {list{str}} -- list of output files to be parsed, as filenames
        incl_list {list{str}} -- list of method labels to include
        term {str} -- Variable with which to measure learning progress. Accepts one of ['Reward', 'Success', 'Steps-Taken']
    
    Returns:
        means_dict {dict{str, list{float}}} -- dict of lists of values of the term, averaged over all runs for each label. The key is the label
        maxes_dict {dict{str, list{float}}} -- dict of lists of values of the term, maximum over all runs for each label. The key is the label
        mins_dict {dict{str, list{float}}} -- dict of lists of values of the term, minimum over all runs for each label. The key is the label
        values_dict {dict{str, list{list{float}}}} -- dict of lists of values of the term, list of values for each run for each label. The key is the label
    """
    label_count = dict()
    coll = dict()
    episode_coll = dict()
    for fname in files:
        print(fname)
        # f = fname.split('.')
        if 'ic3net' in fname and not 'tar' in fname:
            label = 'IC3Net'
        elif 'dec_tarmac' in fname:
            label = 'Dec-TarMAC'
        elif 'tar_commnet' in fname:
            label = 'Tar-CommNet'
        elif 'commnet' in fname:
            label = 'CommNet'
        elif 'tar_ic3net' in fname:
            if 'jagc' in fname or 'cave' in fname:
                label = 'TarMAC+CAVE'
            else:
                label = 'TarMAC'
        elif 'gacomm' in fname:
            if 'jagc' in fname or 'cave' in fname:
                label = 'GA-Comm+CAVE'
            else:
                label = 'GA-Comm'
        elif ('gcomm' in fname or 'magic' in fname) and not 'complete' in fname:
            if 'jagc' in fname or 'cave' in fname:
                label = 'MAGIC+CAVE'
            else:
                label = 'MAGIC'
        elif 'gcomm' in fname and 'complete' in fname:
            label = 'MAGIC w/o the Scheduler'
        else:
            # raise ValueError("Cannot find a label for the file {}".format(fname))
            print("Cannot find a label for the file {}".format(fname))
            continue
        
        if label not in incl_list:
            continue
        
        if label not in coll:
            coll[label] = []
            episode_coll[label] = []

        coll[label] = read_file(coll[label], fname, term)
        episode_coll[label] = read_file(episode_coll[label], fname, 'Episode')

    values_dict = {}
    means_dict = {}
    maxes_dict = {}
    mins_dict = {}
    for label in coll.keys():
        all_values = []
        mean_values = []
        max_values = []
        min_values = []

        for val in coll[label]:
            all_values.append(val)

            mean = sum(val) / len(val)

            if term == 'Success':
                mean *= 100
            mean_values.append(mean)
            variance = np.std(val)/(np.sqrt(len(val)))

            if term == 'Success':
                variance *= 100
            variance = variance if variance < 20 else 20
            max_values.append(mean + variance)
            min_values.append(mean - variance)
        
        # if end == 0:
        #     end = len(mean_values)
        
        values_dict[label] = all_values #[start:end]
        means_dict[label] = mean_values #[start:end]
        maxes_dict[label] = max_values #[start:end]
        mins_dict[label] = min_values #[start:end]
    
    return means_dict, maxes_dict, mins_dict, values_dict

if __name__ == "__main__":
    in_dir = sys.argv[1]
    output_name = str(sys.argv[2])
    term = str(sys.argv[3])
    # Not going to bother with start and end since I can deal with that when reading the csv
    # if len(sys.argv) == 5:
    #     start = 0
    #     end = int(sys.argv[4])
    # elif len(sys.argv) == 6:
    #     start = int(sys.argv[4])
    #     end = int(sys.argv[5])
    # else:
    #     start = 0
    #     end = 0

    # List of methods to include, default: all
    # incl_list = method_list
    incl_list = ['CommNet', 'IC3Net', 'TarMAC', 'TarMAC+CAVE', 'GA-Comm', 'GA-Comm+CAVE', 'MAGIC', 'MAGIC+CAVE'] #, 'GA-Comm', 'GA-Comm+CAVE', 'MAGIC', 'MAGIC+CAVE'] #, 'Dec-TarMAC'] #, 'Tar-CommNet', 'MAGIC w/o the Scheduler'

    if os.path.isdir(in_dir):
        # Collect in_dir and find all the data_files to plot
        files, _ = find_data_files(in_dir)# + "*")
        save_dir = in_dir
    else:
        # Change variables so we can plot a single file
        files = [in_dir]
        path_index = in_dir.rfind("/") + 1
        in_dir = in_dir[:path_index]
        save_dir = in_dir + 'individual_plots'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # Collect the data
    means_dict, maxes_dict, mins_dict, values_dict = parse_data(files, incl_list, term) #, start, end)

    main_df = pd.DataFrame()
    for x_label in means_dict.keys():
        print('x label:', x_label)
        # Save data to a pandas DataFrame
        print(len(means_dict[x_label]))
        main_df[x_label+': mean'] = means_dict[x_label]
        main_df[x_label+': max'] = maxes_dict[x_label]
        main_df[x_label+': min'] = mins_dict[x_label]
        main_df[x_label+': values'] = values_dict[x_label]
        
        # # Collect significance tests by epoch
        # for y_label in means_dict.keys():
        #     print('y label:', y_label)
        #     if x_label == y_label:
        #         continue

        #     t_stat_list = []
        #     t_p_value_list = []
        #     wilc_stat_list = []
        #     wilc_p_value_list = []
        #     for idx in range(len(means_dict[x_label])): # epochs
        #         # Apply the independent t-test
        #         t_stat, p_value = ttest_ind(values_dict[x_label][idx], values_dict[y_label][idx], alternative='greater')
        #         t_stat_list.append(t_stat)
        #         t_p_value_list.append(p_value)
        #         # print(f"t-Test(statistic={t_stat}, p-value={p_value})")

        #         # Apply the Wilcoxon signed-rank test
        #         if np.all((np.array(values_dict[x_label][idx]) - np.array(values_dict[y_label][idx])) == 0):
        #             wilc_stat_list.append(None)
        #             wilc_p_value_list.append(1.)
        #         else:
        #             wilc_res = wilcoxon(values_dict[x_label][idx], values_dict[y_label][idx], alternative='greater')
        #             wilc_stat_list.append(wilc_res.statistic)
        #             wilc_p_value_list.append(wilc_res.pvalue)
        #             # print(wilc_res)
            
        #     main_df[x_label+' vs '+y_label+': t_stat'] = t_stat_list
        #     main_df[x_label+' vs '+y_label+': t_p_value'] = t_p_value_list
        #     main_df[x_label+' vs '+y_label+': wilcoxon_stat'] = wilc_stat_list
        #     main_df[x_label+' vs '+y_label+': wilcoxon_p_value'] = wilc_p_value_list
    
    # Save the plot
    if ".csv" not in output_name:
        output_name = output_name + ".csv"
    main_df.to_csv(str(save_dir) + "/" + output_name)


    
    # # This version did tests over whole learning curves
    # for x_label in means_dict.keys():
    #     for y_label in means_dict.keys(): #j in range(len(incl_list)):
    #         if x_label == y_label:
    #             continue

    #         print('\n')
    #         print(x_label, "vs.", y_label)

    #         # ## Mean ##
    #         # print("Mean")
    #         # # Apply the independent t-test
    #         # t_stat, p_value = ttest_ind(means_dict[x_label], means_dict[y_label], alternative='greater')
    #         # print(f"t-Test(statistic={t_stat}, p-value={p_value})")

    #         # # Apply the Wilcoxon signed-rank test
    #         # res = wilcoxon(means_dict[x_label], means_dict[y_label], alternative='greater')
    #         # print(res)
            
    #         # ## All Values ##
    #         # print()
    #         # print("All Values")
    #         # Apply the independent t-test
    #         t_stat, p_value = ttest_ind(values_dict[x_label], values_dict[y_label], alternative='greater')
    #         print(f"t-Test(statistic={t_stat}, p-value={p_value})")

    #         # Apply the Wilcoxon signed-rank test
    #         res = wilcoxon(values_dict[x_label], values_dict[y_label], alternative='greater')
    #         print(res)