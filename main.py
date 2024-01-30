import pandas as pd
import numpy as np

df = pd.read_csv('age_data.csv')

df['Class'] = df['Class'].map({'No': 0, 'Yes': 1})

def calculate_entropy(data):
    classes, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(data, threshold):
    subset1 = data[data['Age'] <= threshold]['Class']
    subset2 = data[data['Age'] > threshold]['Class']

    entropy_subset1 = calculate_entropy(subset1)
    entropy_subset2 = calculate_entropy(subset2)

    total_entropy = calculate_entropy(data['Class'])
    info_gain = total_entropy - (len(subset1) / len(data) * entropy_subset1 + len(subset2) / len(data) * entropy_subset2)

    return info_gain

optimal_threshold = None
max_info_gain = -1

info_gains = []

for threshold in df['Age'].unique():
    info_gain = calculate_information_gain(df, threshold)
    info_gains.append(info_gain)

    if info_gain > max_info_gain:
        max_info_gain = info_gain
        optimal_threshold = threshold

for i, threshold in enumerate(df['Age'].unique()):
    print(f"Threshold: {threshold}, Information Gain: {info_gains[i]}")

print("\nOptimal Threshold:", optimal_threshold)
print("Corresponding Information Gain:", max_info_gain)
