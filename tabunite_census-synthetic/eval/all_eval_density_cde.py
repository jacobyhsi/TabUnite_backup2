import os
import re
import numpy as np
from collections import defaultdict

# Function to extract the CDE value from a file
def extract_cde_value(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(r'CDE: ([0-9.]+)', content)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"CDE value not found in {file_path}")

# Dictionary to store the file paths by method
file_paths_by_method = defaultdict(list)
methods = ["i2bflow", "dilflow", "oheflow", "tabflow", "i2bddpm", "dicddpm", "oheddpm", "tabddpm"]
methods = ["oheflow", "i2bddpm", "dicddpm", "oheddpm", "tabddpm"]


# Populate the dictionary with file paths
base_path = "/voyager/projects/jacobyhsi/tabunite_census-synthetic/eval/density/syn1/"
for method in methods:
    for num_train in range(1, 2):
        for num_sample in range(1, 4):
            file_name = f"quality_cde_{method}_syn1_{num_train}-{num_sample}.txt"
            file_path = os.path.join(base_path, method, file_name)
            file_paths_by_method[method].append(file_path)

# Calculate mean and standard deviation for each method
results = {}
for method, file_paths in file_paths_by_method.items():
    cde_values = []
    for file_path in file_paths:
        try:
            cde_value = extract_cde_value(file_path)
            cde_values.append(cde_value)
        except (ValueError, FileNotFoundError) as e:
            print(e)
            continue
    mean_cde = np.mean(cde_values)
    std_cde = np.std(cde_values)
    results[method] = {'mean': mean_cde, 'std': std_cde}

# Print the results with mean in percentage format and sd with three significant figures
for method, stats in results.items():
    mean_percentage = stats['mean'] * 100
    std_percentage = stats['std'] * 100
    print(f"Method: {method}")
    print(f"  Mean CDE: {mean_percentage:.2f}")
    print(f"  Standard Deviation of CDE: {std_percentage:.3f}")
