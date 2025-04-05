import numpy as np
from scipy import stats
from scipy.stats import entropy
import pandas as pd
import os


data_path = os.path.join(os.getcwd(), 'data', 'tennisds.csv')
print(data_path)

if os.path.isfile(data_path):
    data = pd.read_csv(data_path)

    target_counts = data['Play Tennis'].value_counts(normalize=True)
    total_entropy = entropy(target_counts, base=2)
    print(f"entropy total is: {total_entropy:.3f}\n")

    attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    information_gains = {}

    for attr in attributes:
        print(f"\n--- Attribute: {attr}")
        remainder = 0
        for val, group in data.groupby(attr):
            label_counts = group['Play Tennis'].value_counts(normalize=True)
            probs = [label_counts.get('Yes', 0), label_counts.get('No', 0)]
            group_entropy = entropy(probs, base=2)
            print(f"{val}: P(Yes) = {probs[0]:.3f}, P(No) = {probs[1]:.3f} -> entropy= {group_entropy:.3f}")

            weight = len(group) / len(data)
            remainder += weight * group_entropy

        info_gain = total_entropy - remainder
        information_gains[attr] = info_gain
        print(f"{attr}: Info Gain = {info_gain:.3f}\n")

    best_attr = max(information_gains, key=information_gains.get)
    print(f"\nBest attribute to split on: {best_attr}")

else:
    raise FileNotFoundError("csv file not found")


