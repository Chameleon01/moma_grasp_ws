import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_array(x):
    # substitue one ore more consecutive spaces with a dot
    number_string = x.strip('[]')
    array = np.fromstring(number_string, sep=' ')
    return array

model_idxs_init = [3, 7, 8, 11, 18, 19,20, 28,32, 41, 44, 45, 48, 49, 50]
models_names = ['051_large_clamp', '065-i_cups', '048_hammer', '033_spatula', '037_scissors', '063-a_marbles', '072-k_toy_airplane', '031_spoon', '044_flat_screwdriver', '019_pitcher_base', '035_power_drill', '032_knife', '006_mustard_bottle', '025_mug', '043_phillips_screwdriver']
print(len(model_idxs_init))
df_to_export = pd.DataFrame({"model_idx":model_idxs_init, "model_name":models_names})
df_to_export.to_csv("model_idx_to_name.csv", index=False)  

# convert model_idxs_init and models_names to dictionary
model_dict = dict(zip(model_idxs_init, models_names))


# Load data
data_over_runs = {"run":[],"mode":[], "grasped_model_idx":[],"avg_quality":[], "avg_quality_arr_len": [], "avg_max_quality":[], "avg_min_quality":[], "avg_std_quality":[], "grasp_success_count":[]}
for run in range(1,47):
    data = pd.read_csv(f"data/eval_data_{run}.csv")
    data['quality_arr'] = data['quality_arr'].apply(clean_array)
    # columns: mode,quality_arr,model_idx,min_quality,max_quality,mean_quality,std_quality,grasp_success
    for mode in [0,1,2]:
        #select data for each mode
        mode_data = data[data['mode'] == mode]
        # Display shapes of the data
        print(mode_data.shape)

        # Display mean of 'mean_quality' columns
        print(f"Mode {mode} meand quality: {mode_data['mean_quality'].mean()}")
        print(f"Mode {mode} max quality: {mode_data['max_quality'].mean()}")
        print(f"Mode {mode} min quality: {mode_data['min_quality'].mean()}")
        print(f"Mode {mode} std quality: {mode_data['std_quality'].mean()}")
        print(f"Mode {mode} grasp success count: {mode_data['grasp_success'].sum()}")

        print(f"Mode {mode} quality_arr lenght: {mode_data['quality_arr'].apply(lambda x: x.size).mean()}")
        data_over_runs["mode"].append(mode)
        data_over_runs["avg_quality"].append(mode_data['mean_quality'].mean())
        data_over_runs["avg_max_quality"].append(mode_data['max_quality'].mean())
        data_over_runs["avg_min_quality"].append(mode_data['min_quality'].mean())
        data_over_runs["avg_std_quality"].append(mode_data['std_quality'].mean())
        data_over_runs["grasp_success_count"].append(mode_data['grasp_success'].sum())
        data_over_runs["avg_quality_arr_len"].append(mode_data['quality_arr'].apply(lambda x: x.size).mean())

        data_over_runs["grasped_model_idx"].append(mode_data[mode_data["grasp_success"]==1]["model_idx"].tolist())
        data_over_runs["run"].append(run)

# print for each data over runs
data_over_runs = pd.DataFrame(data_over_runs)

data_mode_0 = data_over_runs[data_over_runs['mode'] == 0]
data_mode_1 = data_over_runs[data_over_runs['mode'] == 1]    
data_mode_2 = data_over_runs[data_over_runs['mode'] == 2]

print(data_mode_0["avg_max_quality"].mean())
print(data_mode_1["avg_max_quality"].mean())
print(data_mode_2["avg_max_quality"].mean())

print(data_mode_0["grasp_success_count"].mean())
print(data_mode_1["grasp_success_count"].mean())
print(data_mode_2["grasp_success_count"].mean())

print(data_mode_0["avg_quality_arr_len"].mean())
print(data_mode_1["avg_quality_arr_len"].mean())
print(data_mode_2["avg_quality_arr_len"].mean())

# analyse model idx grasp success count for each mode

all_grasped_idx_mode_0 = data_mode_0["grasped_model_idx"].tolist()
all_grasped_idx_mode_0 = [item for sublist in all_grasped_idx_mode_0 for item in sublist]

all_grasped_idx_mode_1 = data_mode_1["grasped_model_idx"].tolist()
all_grasped_idx_mode_1 = [item for sublist in all_grasped_idx_mode_1 for item in sublist]

all_grasped_idx_mode_2 = data_mode_2["grasped_model_idx"].tolist()
all_grasped_idx_mode_2 = [item for sublist in all_grasped_idx_mode_2 for item in sublist]


all_model_idx = all_grasped_idx_mode_0 + all_grasped_idx_mode_1 + all_grasped_idx_mode_2
all_model_idx = list(set(all_model_idx))    
print("Mode 0")
a = np.array(all_grasped_idx_mode_0)
unique_0, counts_0 = np.unique(a, return_counts=True)

print("Mode 1")
a = np.array(all_grasped_idx_mode_1)
unique_1, counts_1 = np.unique(a, return_counts=True)

print("Mode 2")
a = np.array(all_grasped_idx_mode_2)
unique_2, counts_2 = np.unique(a, return_counts=True)

mode_0 = []
mode_1 = []
mode_2 = []
model_idxs = []
for idx in all_model_idx:
    print(f"Model idx {idx} is grasped {counts_0[unique_0 == idx]} times in mode 0")
    print(f"Model idx {idx} is grasped {counts_1[unique_1 == idx]} times in mode 1")
    print(f"Model idx {idx} is grasped {counts_2[unique_2 == idx]} times in mode 2")
    counts0 = counts_0[unique_0 == idx]
    counts1 = counts_1[unique_1 == idx]
    counts2 = counts_2[unique_2 == idx]
    if len(counts0) == 0:
        counts0 = 0
    else:
        counts0 = counts0[0]

    if len(counts1) == 0:
        counts1 = 0
    else:
        counts1 = counts1[0]

    if len(counts2) == 0:
        counts2 = 0
    else:
        counts2 = counts2[0]

    mode_0.append(counts0)
    mode_1.append(counts1)
    mode_2.append(counts2)
    model_idxs.append(idx)

width = 0.2

# Create x positions for each group
x = np.arange(len(model_idxs))  # integer x positions for the bars
model_idxs_0 = x
model_idxs_1 = x + width
model_idxs_2 = x - width

# Create the plot
# set size of the plot
fig, ax = plt.subplots()
ax.bar(model_idxs_0, mode_0, width=width, label='Best view')
ax.bar(model_idxs_1, mode_1, width=width, label='Static view')
ax.bar(model_idxs_2, mode_2, width=width, label='Random view')

# Add labels, title, and legend
ax.set_xlabel('Model Index')
ax.set_ylabel('Number of successful grasps')
ax.set_title('Total successful grasps per model across all runs')

# Set x-ticks to the middle of the groups
ax.set_xticks(x)
ax.set_xticklabels(model_idxs)  # Set actual model indices as x-tick labels
# ax.set_xticklabels([model_dict[idx] for idx in model_idxs])  # Set actual model indices as x-tick labels

ax.autoscale(tight=False)
ax.legend()

#set figure size with width as an A4 paper
fig.set_size_inches(9.7, 5.6)

# Show the plot
plt.savefig("figures/grasp_success_count.png")
plt.show()


# plot quality array length for each mode
data = {
    'Best View': data_mode_0["avg_quality_arr_len"].tolist(),
    'Random View': data_mode_2["avg_quality_arr_len"].tolist(),
    'Static View': data_mode_1["avg_quality_arr_len"].tolist(),
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate summary statistics
summary_stats = df.describe()

# Extract min, max, mean for overlaying on the boxplot
mins = summary_stats.loc['min']
maxs = summary_stats.loc['max']
means = summary_stats.loc['mean']

### Step 2: Create the Boxplot

# Plotting the boxplot
ax = df.boxplot(grid=False, return_type='dict')

### Step 3: Overlay Mean and Min-Max Indicators

# Overlay means as scatter points
for i, mean in enumerate(means, start=1):
    plt.scatter(i, mean, color='red', zorder=3)

# Draw min-max ranges
for i, (min_val, max_val) in enumerate(zip(mins, maxs), start=1):
    plt.plot([i, i], [min_val, max_val], color='blue', linestyle='-', marker='_', markersize=10)

#set siye of the plot
plt.gcf().set_size_inches(9.7, 5.6)

# Adding labels and title
plt.xlabel('Policy')
plt.ylabel('Quality Tensor Size')
plt.title('Quality Tensor Size by Policy')
plt.suptitle('')  # Clear the automatic suptitle that Pandas might add
plt.savefig("figures/quality_array_length_stats.png")
plt.show()


# plot grasping count for each mode
data = {
    'Best View': data_mode_0["grasp_success_count"].tolist(),
    'Static View': data_mode_1["grasp_success_count"].tolist(),
    'Random View': data_mode_2["grasp_success_count"].tolist(),
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate summary statistics
summary_stats = df.describe()

# Extract min, max, mean for overlaying on the boxplot
mins = summary_stats.loc['min']
maxs = summary_stats.loc['max']
means = summary_stats.loc['mean']

### Step 2: Create the Boxplot

# Plotting the boxplot
ax = df.boxplot(grid=False, return_type='dict')

### Step 3: Overlay Mean and Min-Max Indicators

# Overlay means as scatter points
for i, mean in enumerate(means, start=1):
    plt.scatter(i, mean, color='red', zorder=3)

# Draw min-max ranges
for i, (min_val, max_val) in enumerate(zip(mins, maxs), start=1):
    plt.plot([i, i], [min_val, max_val], color='blue', linestyle='-', marker='_', markersize=10)

plt.gcf().set_size_inches(9.7, 5.6)

# Adding labels and title
plt.xlabel('Policy')
plt.ylabel('Number of successful grasps')
plt.title('Number of successful grasps by policy per run')
plt.suptitle('')  # Clear the automatic suptitle that Pandas might add
plt.savefig("figures/grasp_success_count_stats.png")
plt.show()