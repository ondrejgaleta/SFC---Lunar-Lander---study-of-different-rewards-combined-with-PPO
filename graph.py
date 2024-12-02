# Script to create graphs used in documentation
# Date: 1th of December 2024
# Author: Ondrej Galeta 
# Coded with a very extensive usage of ChatGPT

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse

# Parse folder argument with default as current directory
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", help="Folder containing CSV files", default='./data')
args = parser.parse_args()

# Get CSV files
folder_path = args.data_folder
if not os.path.isdir(folder_path):
    raise ValueError(f"Folder '{folder_path}' does not exist.")
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Folder containing CSV files
folder_path = './data/'  # Replace with your folder path
csv_files = glob.glob(f"{folder_path}/*.csv")

file_groups = {}
for file in csv_files:
    prefix = '_'.join(os.path.basename(file).split('_')[:-1])
    if prefix not in file_groups:
        file_groups[prefix] = []
    file_groups[prefix].append(file)

# Initialize the plot
plt.figure(figsize=(12, 8))

# Loop through each group
for group, files in file_groups.items():
    group_data = []
    
    # Loop through the files in the group and read the data
    for file in files:
        data = pd.read_csv(file)
        group_data.append(data)
    
    # Concatenate all the dataframes in the group
    group_df = pd.concat(group_data, ignore_index=True)

    # Truncate rows based on the smallest file length
    min_length = min(len(df) for df in group_data)
    group_df = group_df.iloc[:min_length]
    
    # Ensure there are no NaN values in the relevant column
    group_df = group_df.dropna(subset=[group_df.columns[2]])
    
    # Compute the cumulative mean of the third column for the entire group
    group_df['Cumulative_Mean'] = group_df.iloc[:, 2].expanding().mean()

    # Skip the first 30 rows
    group_df = group_df.iloc[30:]
    
    # Plot the first column against the cumulative mean for the group
    plt.plot(
        group_df.iloc[:, 0], 
        group_df['Cumulative_Mean'], 
        label=f"{group}"  # Use the group name as label
    )

# Customize the plot
plt.xlabel('Episode')
plt.ylabel('Cumulative Mean of Baseline Reward')
plt.legend(title="Reward functions")
plt.grid(True)
plt.savefig("graph_grouped_cumulative_mean.png", dpi=300, bbox_inches='tight')
plt.show()


# Initialize the plot for the second plot
plt.figure(figsize=(12, 8))

# Loop through each group
for group, files in file_groups.items():
    group_data = []
    
    # Loop through the files in the group and read the data
    for file in files:
        data = pd.read_csv(file)
        group_data.append(data)
    
    # Concatenate all the dataframes in the group
    group_df = pd.concat(group_data)

    # Truncate rows based on the smallest file length
    min_length = min(len(df) for df in group_data)
    group_df = group_df.iloc[:min_length]
    
    # Compute the count of values > 150 in the last 10 rows using a rolling window
    group_df['Count_Last_100'] = group_df.iloc[:, 2].rolling(window=100).apply(lambda x: (x > 150).sum(), raw=True)
    
    # Skip the first 30 rows
    group_df = group_df.iloc[30:]

    # Plot the first column against the calculated count 
    plt.plot(
        group_df.iloc[:, 0], 
        group_df['Count_Last_100']/100, 
        label=f"{group}"  # Use the group name as label
    )

# Customize the plot for second graph
plt.xlabel('Episode')
plt.ylabel('Successful Landings Ratio')
plt.legend(title="Reward functions")
plt.grid(True)
plt.savefig("graph_grouped_count.png", dpi=300, bbox_inches='tight')
plt.show()


# Initialize the plot for the third plot
plt.figure(figsize=(12, 8))

# Loop through each group
for group, files in file_groups.items():
    group_data = []
    
    # Loop through the files in the group and read the data
    for file in files:
        data = pd.read_csv(file)
        group_data.append(data)
    
    # Concatenate all the dataframes in the group
    group_df = pd.concat(group_data)

    # Truncate rows based on the smallest file length
    min_length = min(len(df) for df in group_data)
    group_df = group_df.iloc[:min_length]

    
    # Compute the cumulative mean of the third column
    group_df['Cumulative_Mean'] = group_df.iloc[:, 2].expanding().mean()
    
    
    # Compute the cumulative sum of the fourth column for the x-axis
    group_df['Cumulative_Sum'] = group_df.iloc[:, 3].cumsum()



    # Plot the cumulative sum against the cumulative mean for the group
    plt.plot(
        group_df['Cumulative_Sum'], 
        group_df['Cumulative_Mean'], 
        label=f"{group}"  # Use the group name as label
    )

# Customize the plot for third graph
plt.xlabel('Cumulative Sum of steps')
plt.ylabel('Cumulative Mean of Baseline Reward')
plt.legend(title="Reward functions")
plt.grid(True)
plt.savefig("graph_grouped_cumulative_sum.png", dpi=300, bbox_inches='tight')
plt.show()
