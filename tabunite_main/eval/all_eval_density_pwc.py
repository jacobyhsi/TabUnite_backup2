import os
import re
import numpy as np
from collections import defaultdict

# Function to extract the PWC value from a file
def extract_pwc_value(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(r'PWC: ([0-9.]+)', content)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"PWC value not found in {file_path}")

# Dictionary to store the file paths by method
file_paths_by_method = defaultdict(list)
methods = ["i2bflow", "dilflow", "oheflow", "tabflow", "i2bddpm", "dicddpm", "oheddpm", "tabddpm"]
methods = ["oheddpm"]


# Populate the dictionary with file paths
base_path = "tabsyn_syn/eval/density/syn1/"
for method in methods:
    for num_train in range(1, 2):
        for num_sample in range(1, 4):
            file_name = f"quality_pwc_{method}_syn1_{num_train}-{num_sample}.txt"
            file_path = os.path.join(base_path, method, file_name)
            file_paths_by_method[method].append(file_path)

# Calculate mean and standard deviation for each method
results = {}
for method, file_paths in file_paths_by_method.items():
    pwc_values = []
    for file_path in file_paths:
        try:
            pwc_value = extract_pwc_value(file_path)
            pwc_values.append(pwc_value)
        except (ValueError, FileNotFoundError) as e:
            # Handle both ValueError (from extract_pwc_value) and FileNotFoundError
            print(e)
            continue
    if pwc_values:  # Only calculate if there are valid PWC values
        mean_pwc = np.mean(pwc_values)
        std_pwc = np.std(pwc_values)
        results[method] = {'mean': mean_pwc, 'std': std_pwc}

# Print the results with mean in percentage format and sd with three significant figures
for method, stats in results.items():
    mean_percentage = stats['mean'] * 100
    std_percentage = stats['std'] * 100
    print(f"Method: {method}")
    print(f"  Mean PWC: {mean_percentage:.2f}%")
    print(f"  Standard Deviation of PWC: {std_percentage:.3f}%")
