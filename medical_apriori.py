"""
Medical Apriori Algorithm Module for Association Rule Mining.

This module implements the Apriori algorithm for mining association rules
from medical records, specifically for hydrocephalus postoperative infection
prediction.
"""

import itertools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Cartesian:
    """
    Cartesian product generator for combining itemsets.

    This class generates the Cartesian product of multiple data lists,
    used for combining frequent itemsets with class labels.
    """

    def __init__(self):
        self._data_list = []

    def add_data(self, data: List = None):
        """
        Add data list for Cartesian product generation.

        Args:
            data: List of items to add
        """
        if data is None:
            data = []
        self._data_list.append(data)

    def build(self) -> List[Tuple]:
        """
        Compute Cartesian product of all added data lists.

        Returns:
            List of tuples representing the Cartesian product
        """
        return list(itertools.product(*self._data_list))


def load_dataset(filename: str) -> List[List[str]]:
    """
    Load dataset from a CSV file.

    Args:
        filename: Path to the data file

    Returns:
        List of transactions where each transaction is a list of items
    """
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.strip().split(","))
    return data


def count_items(items: List[str], data: List[List[str]]) -> int:
    """
    Count occurrences of an itemset in the dataset.

    Args:
        items: Itemset to count
        data: Dataset to search in

    Returns:
        Number of occurrences
    """
    count = 0
    for transaction in data:
        if set(items).issubset(set(transaction)):
            count += 1
    return count


def get_frequent_items(
    items: List[List[str]], data: List[List[str]]
) -> Tuple[List[List[str]], List[float]]:
    """
    Get frequent items and their support values.

    Args:
        items: Candidate itemsets
        data: Dataset

    Returns:
        Tuple of (frequent_itemsets, support_values)
    """
    freq_list = []
    support_list = []

    for item in items:
        support = count_items(item, data) / float(len(data))
        if support > 0:
            if sorted(item) not in freq_list:
                freq_list.append(sorted(item))
                support_list.append(support)

    return freq_list, support_list


def generate_frequent_itemsets(
    items: List[List[str]], data: List[List[str]]
) -> Tuple[List[List[str]], Dict[frozenset, float]]:
    """
    Generate frequent itemsets from candidate items.

    Args:
        items: Candidate itemsets
        data: Dataset

    Returns:
        Tuple of (frequent_itemsets, support_dict)
    """
    support_dict = {}
    frequent_itemsets = []

    freq_list, support_list = get_frequent_items(items, data)

    for freq, support in zip(freq_list, support_list):
        frequent_itemsets.append(freq)
        support_dict[frozenset(freq)] = support

    return frequent_itemsets, support_dict


def generate_rules(
    frequent_merge: List[Tuple],
    merge_support: List[float],
    left_support: Dict,
    enhancement_ratio: Dict,
    class_support: Dict,
) -> List[Tuple]:
    """
    Generate association rules from frequent itemsets.

    Args:
        frequent_merge: Frequent itemsets merged with class labels
        merge_support: Support values for merged itemsets
        left_support: Support values for left-hand side items
        enhancement_ratio: Enhancement ratio values
        class_support: Support values for class labels

    Returns:
        List of association rules
    """
    rules = []
    for freq, support in zip(frequent_merge, merge_support):
        rules.extend(
            _rule(freq, support, left_support, enhancement_ratio, class_support, [])
        )
    return rules


def _rule(
    freq: Tuple,
    support: float,
    left_support: Dict,
    enhancement_ratio: Dict,
    class_support: Dict,
    current_rules: List,
) -> List[Tuple]:
    """
    Generate a single association rule.

    Args:
        freq: Frequent itemset tuple (LHS, RHS)
        support: Support value
        left_support: Support values for LHS
        enhancement_ratio: Enhancement ratio values
        class_support: Support values for class
        current_rules: Accumulated rules

    Returns:
            Updated list of rules
    """
    lift = class_support[frozenset(freq[0])] / left_support[frozenset(freq[0])]
    strength = (2 * enhancement_ratio[frozenset(freq[0])] * lift) / (
        enhancement_ratio[frozenset(freq[0])] + lift
    )

    if lift >= 1:
        current_rules.append(
            (freq[0], freq[1], enhancement_ratio[frozenset(freq[0])], lift, strength)
        )

    return current_rules


def mine_association_rules(
    data_file: str = "./data/2020.7.11_fenji.xlsx",
    frequent_itemsets_file: str = "./data/fk_left_neg.txt",
    min_enhancement_ratio: float = 1.0,
    min_confidence: float = 0.2,
) -> None:
    """
    Main function to mine association rules from medical data.

    Args:
        data_file: Path to the medical data Excel file
        frequent_itemsets_file: Path to the frequent itemsets file
        min_enhancement_ratio: Minimum enhancement ratio threshold
        min_confidence: Minimum confidence threshold
    """
    # Load data
    df = pd.read_excel(data_file, header=0, dtype=str)

    # Split into positive and negative classes
    df_positive = df[df["47术后是否感染"] == "471"]
    df_negative = df[df["47术后是否感染"] == "472"]

    data_all = df.values.tolist()
    data_positive = df_positive.values.tolist()
    data_negative = df_negative.values.tolist()

    # Load frequent itemsets
    frequent_itemsets = []
    with open(frequent_itemsets_file, "r", encoding="utf-8") as f:
        for line in f:
            frequent_itemsets.append(line.strip().split(" "))

    # Generate frequent itemsets for positive class
    fk_positive, support_positive = generate_frequent_itemsets(
        frequent_itemsets, data_negative
    )
    print(f"Frequent itemsets: {len(support_positive)}")
    for key, value in support_positive.items():
        print(f"{key} : {value:.2f}")

    # Generate frequent itemsets for negative class
    fk_negative, support_negative = generate_frequent_itemsets(fk_positive, data_positive)
    print(f"Frequent itemsets: {len(support_negative)}")
    for key, value in support_negative.items():
        print(f"{key} : {value:.2f}")

    # Filter by enhancement ratio
    fk_final = []
    enhancement_dict = {}

    for item in fk_negative:
        if (
            support_positive[frozenset(item)] / support_negative[frozenset(item)]
            >= min_enhancement_ratio
        ):
            fk_final.append(item)
            enhancement_dict[frozenset(item)] = (
                support_positive[frozenset(item)] / support_negative[frozenset(item)]
            )

    print(f"Frequent itemsets after ER filtering: {len(fk_final)}")
    for key, value in enhancement_dict.items():
        print(f"{key} : {value:.2f}")

    # Generate final frequent itemsets
    fk_all, support_all = generate_frequent_itemsets(fk_final, data_all)
    print(f"Final frequent itemsets: {len(support_all)}")
    for key, value in support_all.items():
        print(f"{key} : {value:.2f}")

    # Final filtering
    fk_final = []
    enhancement_dict = {}

    for item in fk_all:
        if (
            support_positive[frozenset(item)] / support_all[frozenset(item)]
            >= min_enhancement_ratio
        ):
            fk_final.append(item)
            enhancement_dict[frozenset(item)] = (
                support_positive[frozenset(item)] / support_all[frozenset(item)]
            )

    print(f"Final filtered itemsets: {len(fk_final)}")
    for key, value in enhancement_dict.items():
        print(f"{key} : {value:.2f}")

    # Generate Cartesian product with class label
    right = [["472"]]
    cart = Cartesian()
    cart.add_data(fk_final)
    cart.add_data(right)
    issues = cart.build()
    print(f"Total rules to evaluate: {len(issues)}")

    # Calculate support for each rule
    issue_support = []
    for items in issues:
        support = count_items(items[0] + items[1], data_all) / len(data_all)
        issue_support.append(support)

    # Generate association rules
    rules = generate_rules(
        issues, issue_support, support_all, enhancement_dict, support_positive
    )
    print(f"Association rules: {len(rules)}")

    # Extract metrics for visualization
    x = []
    y = []
    c = []
    for item in rules:
        x.append(item[2])  # Enhancement ratio
        y.append(item[3])  # Lift
        c.append(item[4])  # Strength

    # Plot scatter diagram
    plt.style.use("default")
    sc = plt.scatter(x, y, c=c, s=20, marker="o", cmap="Blues")
    plt.xlabel("Enhancement Ratio")
    plt.ylabel("Lift")
    cb = plt.colorbar(sc)
    cb.set_label("Rule Strength", labelpad=-18, y=-0.02, rotation=0)
    plt.savefig("./data/figure/scatter.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    mine_association_rules()
