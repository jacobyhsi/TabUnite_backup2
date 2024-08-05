import os
import re
import numpy as np
from collections import defaultdict

# Function to extract Alpha_Precision_all and Beta_Recall_all values from a file
def extract_values(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        alpha_match = re.search(r'Alpha_Precision_all: ([0-9.]+)', content)
        beta_match = re.search(r'Beta_Recall_all: ([0-9.]+)', content)
        if alpha_match and beta_match:
            return float(alpha_match.group(1)), float(beta_match.group(1))
        else:
            raise ValueError(f"Alpha_Precision_all or Beta_Recall_all value not found in {file_path}")

# Dictionary to store the file paths by method
file_paths_by_method = defaultdict(list)
methods = ["i2bflow", "dilflow", "oheflow", "tabflow", "i2bddpm", "dicddpm", "oheddpm", "tabddpm"]
methods = ["i2bddpm", "oheddpm"]

# Populate the dictionary with file paths
base_path = "tabunite_census-synthetic/eval/quality/syn1/"
for method in methods:
    for num_train in range(2, 3):
        for num_sample in range(1, 4):
            file_name = f"{method}_syn1_{num_train}-{num_sample}.txt"
            file_path = os.path.join(base_path, file_name)
            file_paths_by_method[method].append(file_path)

# Calculate mean and standard deviation for each method
results = {}
for method, file_paths in file_paths_by_method.items():
    alpha_precision_values = []
    beta_recall_values = []
    for file_path in file_paths:
        try:
            alpha_precision, beta_recall = extract_values(file_path)
            alpha_precision_values.append(alpha_precision)
            beta_recall_values.append(beta_recall)
        except (ValueError, FileNotFoundError) as e:
            # Handle both ValueError (from extract_values) and FileNotFoundError
            print(e)
            continue
    if alpha_precision_values and beta_recall_values:  # Only calculate if there are valid values
        mean_alpha_precision = np.mean(alpha_precision_values)
        std_alpha_precision = np.std(alpha_precision_values)
        mean_beta_recall = np.mean(beta_recall_values)
        std_beta_recall = np.std(beta_recall_values)
        results[method] = {
            'mean_alpha_precision': mean_alpha_precision,
            'std_alpha_precision': std_alpha_precision,
            'mean_beta_recall': mean_beta_recall,
            'std_beta_recall': std_beta_recall
        }

# Print the results with mean in percentage format and std with three significant figures
for method, stats in results.items():
    mean_alpha_percentage = stats['mean_alpha_precision'] * 100
    std_alpha_percentage = stats['std_alpha_precision'] * 100
    mean_beta_percentage = stats['mean_beta_recall'] * 100
    std_beta_percentage = stats['std_beta_recall'] * 100
    print(f"Method: {method}")
    print(f"  Mean Alpha_Precision_all: {mean_alpha_percentage:.2f}%")
    print(f"  Standard Deviation of Alpha_Precision_all: {std_alpha_percentage:.3f}%")
    print(f"  Mean Beta_Recall_all: {mean_beta_percentage:.2f}%")
    print(f"  Standard Deviation of Beta_Recall_all: {std_beta_percentage:.3f}%")
