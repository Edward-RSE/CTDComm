"""Takes directory(/ies) and a term as input, gathers all data files and plots data about that term
Arguments:
    in_dir - directory for an individual environment type, containing separate folders for each method
    plot_name - filename for the saved plot
    term - Variable with which to measure learning progress. Accepts one of ['Reward', 'Success', 'Steps-Taken']"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import numpy as np
import sys
import os

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (10,8)

# Why not just use default colors?
# colors_map = {
#     'IC3Net': '#fca503',
#     'CommNet': '#b0b0b0',
#     'TarMAC': '#ff6373',
#     'TarMAC-IC3Net': '#b700ff',
#     'GA-Comm': '#77ab3f',
#     'MAGIC (Our Approach)': '#0040ff'#,
#     # 'MAGIC w/o the Scheduler': '#ff6373',
# }
method_list = ['CommNet', 'IC3Net', 'TarMAC', 'Tar-CommNet', 'TarMAC+CAVE', 'GA-Comm', 'GA-Comm+CAVE', 'MAGIC', 'MAGIC+CAVE', 'Dec-TarMAC'] #, 'MAGIC w/o the Scheduler'
# plot_colors = [c for c in colors.TABLEAU_COLORS]
# colors_map = {method_list[i]: colors.TABLEAU_COLORS[plot_colors[i]] for i in range(len(method_list))}
plot_colors = sns.color_palette('Paired')
colors_map = {'CommNet': plot_colors[1], 'IC3Net': plot_colors[-3], 'TarMAC': plot_colors[4], 'TarMAC+CAVE': plot_colors[5],
             'GA-Comm': plot_colors[2], 'GA-Comm+CAVE': plot_colors[3], 'MAGIC': plot_colors[6], 'MAGIC+CAVE': plot_colors[7]}
style_map = {'CommNet': 'solid', 'IC3Net': 'dotted', 'TarMAC': 'dashed', 'TarMAC+CAVE': 'dashdot',
             'GA-Comm': 'dashed', 'GA-Comm+CAVE': 'dashdot', 'MAGIC': (0, (5, 1)), 'MAGIC+CAVE': (0, (3, 1, 1, 1))}


def read_file(vec, file_name, term):
    print(file_name)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return vec

        mean_reward = False
        for idx, line in enumerate(lines):
            if term not in line:
                continue
            epoch_idx = idx
            epoch_line = line
            while 'Epoch' not in epoch_line:
                epoch_idx -= 1
                epoch_line = lines[epoch_idx]

            epoch = int(epoch_line.split(' ')[1].split('\t')[0])
            # if file_name == 'log_files/tj_medium/commnet_tj_medium_no_cur_run1.log':
            #     epoch -= 4000

            floats = line.split('\t')[0]
            left_bracket = floats.find('[')
            right_bracket = floats.find(']')

            if left_bracket == -1 and left_bracket == -1:

                floats = line.split('\t')[0]
                if epoch > len(vec):
                    vec.append([float(floats.split(' ')[-1].strip())])
                else:
                    vec[epoch - 1].append(float(floats.split(' ')[-1].strip()))

            else:
                floats = np.fromstring(floats[left_bracket + 1:right_bracket], dtype=float, sep=' ')

                if epoch > len(vec):
                    vec.append([floats.mean()])
                else:
                    vec[epoch - 1].append(floats.mean())

    return vec

def parse_plot(files, incl_list, term='Reward', window_width=1):
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

        # if 'ic3net_pp_hard' in fname and not 'tar' in fname and term == 'Steps-Taken':
        # 	term = 'Steps-taken'

        coll[label] = read_file(coll[label], fname, term)
        episode_coll[label] = read_file(episode_coll[label], fname, 'Episode')

        # if 'ic3net_pp_hard' in fname and not 'tar' in fname and term == 'Steps-taken':
        # 	term = 'Steps-Taken'

    for label in coll.keys():
        mean_values = []
        max_values = []
        min_values = []

        mean_windowed = []
        max_windowed = []
        min_windowed = []

        i = 1 #this makes the windowing calculations more readable
        for val in coll[label]:
            # if i > 3500: break #for limiting the number of epochs we plot
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

            if i >= window_width:
                mean_windowed.append(sum(mean_values[i-window_width:i]) / window_width)
                max_windowed.append(sum(max_values[i-window_width:i]) / window_width)
                min_windowed.append(sum(min_values[i-window_width:i]) / window_width)

            i += 1

        # mean_episodes = []
        # for epi_val in episode_coll[label]:
        #     mean_episodes.append(sum(epi_val) / len(epi_val))

        print(label)
        print('max: ', np.max(mean_values))
        print('min: ', np.min(mean_values))
        max_idx = np.argmax(mean_values)
        min_idx = np.argmin(mean_values)
        print('max std: ', np.std(coll[label][max_idx]))
        print('min std: ', np.std(coll[label][min_idx]))

        # Original plotting (without windowing)
        # plt.plot(np.arange(len(coll[label])), mean_values, linewidth=2.0, label=label, color=colors_map[label], linestyle=style_map[label])
        # plt.fill_between(np.arange(len(coll[label])), min_values, max_values, color=colors.to_rgba(colors_map[label], alpha=0.2))

        # Plot with windowing
        plt.plot(np.arange(window_width-1, len(coll[label])), mean_windowed, linewidth=2.0, label=label, color=colors_map[label], linestyle=style_map[label])
        plt.fill_between(np.arange(window_width-1, len(coll[label])), min_windowed, max_windowed, color=colors.to_rgba(colors_map[label], alpha=0.2))

        # # Plot with windowing, limit to 3500 epochs
        # plt.plot(np.arange(window_width-1, 3500), mean_windowed, linewidth=2.0, label=label, color=colors_map[label], linestyle=style_map[label])
        # plt.fill_between(np.arange(window_width-1, 3500), min_windowed, max_windowed, color=colors.to_rgba(colors_map[label], alpha=0.2))

        # Unwindowed plot with thinner mean line
        # plt.plot(mean_episodes, mean_values, linewidth=1.5, label=label, color=colors_map[label], linestyle=style_map[label])
        # plt.fill_between(mean_episodes, min_values, max_values, color=colors.to_rgba(colors_map[label], alpha=0.2))

    plt.xlabel('Epochs')
    if term == 'Success':
        term = 'Success Rate (%)'
    plt.ylabel(term)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., prop={'size': 17})
    # plt.legend(framealpha=1)
    plt.grid()
#     plt.title('GFootball {} {}'.format(sys.argv[2], term))

def find_data_files(in_dir):
    """Recursive, list all directories or data files in a given directory
        Input:
            in_dir - directory containing all the data we wish to plot
        Output:
            files - list of directories and data files in in_dir
            depth - number of nested directories between in_dir and the deepest data file"""
    
    # Collect files and directories
    depth = 0
    files = []
    for f in os.listdir(in_dir):
        f_name = in_dir + '/' + f
        
        if os.path.isdir(f_name):
            # For directories, recurse and add the output to files
            sub_files, sub_depth = find_data_files(f_name)
            if sub_files != []:
                files += sub_files
                
                if sub_depth+1 > depth:
                    depth = sub_depth+1
        
        elif 'log' in f or 'slurm' in f:
            # For data files, add it to files
            files.append(f_name)
    
    return files, depth

if __name__ == "__main__":
    # files = glob.glob(sys.argv[1] + "*")
    # # filter out files with ".pt"
    # files = list(filter(lambda x: x.find(".pt") == -1, files))

    # # 'Epoch'/ 'Steps-taken'
    # term = sys.argv[3]
    # parse_plot(files, term)
    # # plt.show()

    # # Saving rather than showing
    # path_index = files[0].rfind("/") + 1
    # plot_file = files[0][:path_index]
    # plot_file = plot_file + str(sys.argv[2]) + ".png"
    # plt.savefig(plot_file)

    in_dir = sys.argv[1]
    plot_name = str(sys.argv[2])
    term = str(sys.argv[3])
    if len(sys.argv) >= 5:
        window_width = int(sys.argv[4])
    else:
        window_width = 1

    # List of methods to include, default: all
    incl_list = method_list
    incl_list = ['CommNet', 'IC3Net', 'TarMAC', 'TarMAC+CAVE'] #, 'GA-Comm', 'GA-Comm+CAVE', 'MAGIC', 'MAGIC+CAVE'] #, 'Dec-TarMAC'] #, 'MAGIC' #, 'Tar-CommNet', 'MAGIC w/o the Scheduler'

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

    # Plot the data
    parse_plot(files, incl_list, term, window_width)

    # Save the plot
    if ".png" not in plot_name:
        plot_name = plot_name + ".png"
    plt.savefig(str(save_dir) + "/" + plot_name)