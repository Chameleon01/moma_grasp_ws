import pandas as pd
import matplotlib.pyplot as plt

# Load data
data_NVB = pd.read_csv("quality_data_1.csv")
data_NoMove = pd.read_csv("quality_data.csv")

# Display shapes of the data
print(data_NVB.shape)
print(data_NoMove.shape)

# Merge data on 'model_idx'
merged_data = pd.merge(data_NVB, data_NoMove, on='model_idx')   
print(merged_data.shape, merged_data.columns)
print(data_NoMove["model_idx"].tolist())

# Display mean of 'mean_quality' columns
print(f"NVB meand quality: {merged_data['mean_quality_x'].mean()}, NoMove meand quality: {merged_data['mean_quality_y'].mean()}")
print(f"NVB max quality: {merged_data['mean_quality_x'].mean()}, NoMove max quality: {merged_data['mean_quality_y'].mean()}")

# Sort data by 'mean_quality_x'
sorted_data = merged_data.sort_values('mean_quality_x')

# Create a figure and axis
fig, ax = plt.subplots()

# Plot mean qualities for both x and y using dots
ax.scatter(sorted_data.index, sorted_data['mean_quality_x'], label='Mean Quality X', color='blue')
ax.scatter(sorted_data.index, sorted_data['mean_quality_y'], label='Mean Quality Y', color='red')

# Labeling
plt.xlabel('Sorted Index')
plt.ylabel('Mean Quality')
plt.title('Comparison of Mean Quality X vs. Mean Quality Y Using Dots')
plt.legend()

# Display the plot
plt.show()

# Save the plot
plt.savefig('mean_quality_comparison_dots.png')
