# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# Assuming 'gpu_utilization.csv' is your file name and it's in the same directory as your script
file_name = 'gpu_utilization.csv'  # Replace with your actual file path

# Read the CSV file into a DataFrame
df = pd.read_csv(file_name)

# Extract each column into a separate array
async_gpu0_utilization = df['async_gpu0_utilization'].to_list()
async_gpu1_utilization = df['async_gpu1_utilization'].to_list()
sync_gpu0_utilization = df['sync_gpu0_utilization'].to_list()
sync_gpu1_utilization = df['sync_gpu1_utilization'].to_list()

# Define the function to plot the utilization
def adjusted_plot_utilization(async_gpu0_utilization, async_gpu1_utilization, sync_gpu0_utilization, sync_gpu1_utilization, title, output_filename):
    plt.style.use('default')  # Resetting style for white background
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    
    # Distinct colors for clarity
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax.plot(async_gpu0_utilization, color=colors[0], label="Async Primary Device", linewidth=1.5)
    ax.plot(async_gpu1_utilization, color=colors[1], label="Async Auxiliary Device", linewidth=1.5)
    ax.plot(sync_gpu0_utilization, color=colors[2], label="Sync Primary Device", linewidth=1.5)
    ax.plot(sync_gpu1_utilization, color=colors[3], label="Sync Auxiliary Device", linewidth=1.5)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(title, fontsize=20, loc='center', pad=10)
    plt.grid(axis='y', alpha=0.5, color='grey')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, handlelength=0.5, fontsize=20)
    y_ticks = [0, 20, 40, 60, 80, 100]
    plt.xlim(0, len(async_gpu0_utilization))
    plt.yticks(y_ticks)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Utilization (%)", fontsize=20)
    plt.tick_params(axis='both', which='both', length=0, labelsize=20)
    plt.subplots_adjust(left=0.1, right=0.92, bottom=0.35, top=0.9)
    plt.savefig(f'{output_filename}')
    plt.show()

# Now we'll call the function to plot the data
adjusted_plot_utilization(
    async_gpu0_utilization, 
    async_gpu1_utilization, 
    sync_gpu0_utilization, 
    sync_gpu1_utilization, 
    "GPU Utilization with asynchronous submission", 
    "GPUUsage.pdf"
)
