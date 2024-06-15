import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

N_RUNS = 46

def clean_array(x):
    """
    Convert a string representation of an array into a numpy array.
    """
    number_string = x.strip('[]')
    array = np.fromstring(number_string, sep=' ')
    return array

def load_data(n_runs, filepath_pattern="data/eval_data_{}.csv"):
    """
    Load and concatenate data from multiple runs.
    """
    all_data = pd.read_csv(filepath_pattern.format(1))
    all_data["run"] = 1

    for run in range(2, n_runs + 1):
        data_run = pd.read_csv(filepath_pattern.format(run))
        data_run["run"] = run
        all_data = pd.concat([all_data, data_run])

    all_data['quality_arr'] = all_data['quality_arr'].apply(clean_array)
    return all_data

def prepare_data_for_analysis(all_data, model_idxs_init, policies):
    """
    Prepare grasp success rates and quality tensor sizes for analysis.
    """
    grasp_success_rates_per_model = {policy: [] for policy in policies}
    quality_tens_size_per_model = {policy: [] for policy in policies}

    for model_idx in model_idxs_init:
        data_model = all_data[all_data["model_idx"] == model_idx]
        for policy in policies:
            # Grasping success
            grasp_successes = data_model[data_model["mode"] == policy]["grasp_success"].tolist()
            grasp_success_rates_per_model[policy].append(sum(grasp_successes) / N_RUNS)

            # Quality tensor sizes
            quality_tens_policy = data_model[data_model["mode"] == policy]["quality_arr"].tolist()
            quality_tens_policy = [len(quality) for quality in quality_tens_policy]
            quality_tens_policy = np.array(quality_tens_policy).mean()
            quality_tens_size_per_model[policy].append(quality_tens_policy)

    return grasp_success_rates_per_model, quality_tens_size_per_model

def prepare_run_based_data(all_data, model_idxs_init, policies):
    """
    Prepare run-based grasp success rates and quality tensor sizes for analysis.
    """
    grasp_success_rates_per_run = {policy: [] for policy in policies}
    quality_tens_size_per_run = {policy: [] for policy in policies}

    for policy in policies:
        grasp_n_policy = all_data[all_data["mode"] == policy]
        grasp_n_policy = grasp_n_policy.groupby("run")["grasp_success"].sum()
        grasp_n_policy = grasp_n_policy / len(model_idxs_init)
        grasp_success_rates_per_run[policy] = grasp_n_policy.tolist()

        quality_tens_size_policy = all_data[all_data["mode"] == policy]
        quality_tens_size_policy["quality_arr"] = quality_tens_size_policy["quality_arr"].apply(lambda x: len(x))
        quality_tens_size_policy = quality_tens_size_policy.groupby("run")["quality_arr"].sum()
        quality_tens_size_policy = quality_tens_size_policy / len(model_idxs_init)
        quality_tens_size_per_run[policy] = quality_tens_size_policy.tolist()

    return grasp_success_rates_per_run, quality_tens_size_per_run

def plot_boxplot(data, ylabel, title, filename):
    """
    Plot a boxplot with summary statistics.
    """
    df = pd.DataFrame(data)
    summary_stats = df.describe()
    summary_stats.to_csv(f"figures/{filename}_stats.csv")

    mins = summary_stats.loc['min']
    maxs = summary_stats.loc['max']
    means = summary_stats.loc['mean']

    ax = df.boxplot(grid=False, return_type='dict')

    for i, mean in enumerate(means, start=1):
        plt.scatter(i, mean, color='red', zorder=3)

    for i, (min_val, max_val) in enumerate(zip(mins, maxs), start=1):
        plt.plot([i, i], [min_val, max_val], color='blue', linestyle='-', marker='_', markersize=10)

    plt.gcf().set_size_inches(9.7, 5.6)
    plt.xticks(ticks=range(1, 4), labels=['Best view', 'Static view', 'Random view'])
    plt.xlabel('Policy')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f"figures/{filename}.png")
    plt.show()

def plot_data_per_model(data_per_model, model_idxs_init, policies_names, ylabel, title, filename):
    """
    Plot the data per model.
    """
    width = 0.2
    x = np.arange(len(model_idxs_init))
    fig, ax = plt.subplots()
    for i, policy in enumerate(policies_names.keys()):
        ax.bar(x + i*width, data_per_model[policy], width=width, label=policies_names[policy])
    ax.set_xlabel('Model Index')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_idxs_init)
    ax.legend()
    fig.set_size_inches(9.7, 5.6)
    plt.savefig(f"figures/{filename}.png")
    plt.show()

def main():
    model_idxs_init = [3, 7, 8, 11, 18, 19,20, 28,32, 41, 44, 45, 48, 49, 50]
    models_names = ['051_large_clamp', '065-i_cups', '048_hammer', '033_spatula', '037_scissors', '063-a_marbles', '072-k_toy_airplane', '031_spoon', '044_flat_screwdriver', '019_pitcher_base', '035_power_drill', '032_knife', '006_mustard_bottle', '025_mug', '043_phillips_screwdriver']
    
    df_to_export = pd.DataFrame({"model_idx":model_idxs_init, "model_name":models_names})
    df_to_export.to_csv("model_idx_to_name.csv", index=False)  

    policies = [0, 1, 2]
    policies_names = {0: "Best view", 1: "Static View", 2: "Random View"}
    all_data = load_data(N_RUNS)
    grasp_success_rates_per_model, quality_tens_size_per_model = prepare_data_for_analysis(all_data, model_idxs_init, policies)
    grasp_success_rates_per_run, quality_tens_size_per_run = prepare_run_based_data(all_data, model_idxs_init, policies)
    plot_data_per_model(grasp_success_rates_per_model, model_idxs_init, policies_names, 'Grasp Success Rate', 'Success Rate per model across all runs', 'grasp_success_rate_per_model')
    plot_data_per_model(quality_tens_size_per_model, model_idxs_init, policies_names, 'Quality Tensor Size', 'Quality Tensor Size per model across all runs', 'quality_tensor_size_per_model')
    plot_boxplot(grasp_success_rates_per_run, 'Grasp Success Rate', 'Grasp Success Rate by policy across all runs', 'grasp_success_rate')
    plot_boxplot(quality_tens_size_per_run, 'Number of grasps above threshold', 'Number of grasps above threshold by policy across all runs', 'quality_tensor_size')

if __name__ == "__main__":
    main()