import numpy as np
from itertools import combinations

# # Example (f variant)
# # 1: 0 6 7 8 9
# # 2: 1 2 3 6 8
# # 3: 5 7
# # 4: 6 7 8
# # 5: 4 6
# # 6: 0 1 2 8
# # 7: 4 7
# # 8: 1 2 9
# # 9: 5 6
# # 10: 6 7 8
data = [[0, 6, 7, 8, 9],
        [1, 2, 3, 6, 8],
        [5, 7],
        [6, 7, 8],
        [4, 6],
        [0, 1, 2, 8],
        [4, 7],
        [1, 2, 9],
        [5, 6],
        [6, 7, 8]]

X_data = np.array(data, dtype=object)

# second_dim = max([elem for row in data for elem in row]) + 1
# X_data = np.zeros((len(data), second_dim))
# for row_ind, row in enumerate(data):
#     for elem in row:
#         X_data[row_ind, elem] = 1


def create_candidate_1(X):
    """
    create the 1-item candidate,
    it's basically creating a frozenset for each unique item
    and storing them in a list
    """
    c1 = []
    for transaction in X:
        for t in transaction:
            t = frozenset([t])
            if t not in c1:
                c1.append(t)

    return c1


def create_freq_item(X, ck, min_support):
    """
    filters the candidate with the specified
    minimum support
    """
    # loop through the transaction and compute
    # the count for each candidate (item)
    item_count = {}
    for transaction in X:
        for item in ck:
            if item.issubset(transaction):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1

    n_row = X.shape[0]
    freq_item = []
    item_support = {}

    # if the support of an item is greater than the
    # min_support, then it is considered as frequent
    for item in item_count:
        support = item_count[item] / n_row
        if support >= min_support:
            freq_item.append(item)

        item_support[item] = support

    return freq_item, item_support


def create_candidate_k(freq_item, k):
    """create the list of k-item candidate"""
    ck = []

    # for generating candidate of size two (2-itemset)
    if k == 0:
        for f1, f2 in combinations(freq_item, 2):
            item = f1 | f2  # union of two sets
            ck.append(item)
    else:
        for f1, f2 in combinations(freq_item, 2):
            # if the two (k+1)-item sets has
            # k common elements then they will be
            # unioned to be the (k+2)-item candidate
            intersection = f1 & f2
            if len(intersection) == k:
                item = f1 | f2
                if item not in ck:
                    ck.append(item)
    return ck


def apriori(X, min_support=3.):
    """
    pass in the transaction data and the minimum support
    threshold to obtain the frequent itemset. Also
    store the support for each itemset, they will
    be used in the rule generation step
    """
    c1 = create_candidate_1(X)
    freq_item, item_support_dict = create_freq_item(X, c1,
                                                    min_support=min_support)

    freq_items = [freq_item]

    k = 0
    while len(freq_items[k]) > 0:
        freq_item = freq_items[k]
        ck = create_candidate_k(freq_item, k)
        freq_item, item_support = create_freq_item(X, ck,
                                                   min_support=min_support)
        freq_items.append(freq_item)
        item_support_dict.update(item_support)
        k += 1

    return freq_items, item_support_dict


def compute_conf(freq_items, item_support_dict, freq_set, subsets,
                 min_confidence):
    """
    create the rules and returns the rules info and the rules's
    right hand side (used for generating the next round of rules)
    if it surpasses the minimum confidence threshold
    """
    rules = []
    right_hand_side = []

    for rhs in subsets:
        lhs = freq_set - rhs
        conf = item_support_dict[freq_set] / item_support_dict[lhs]
        if conf >= min_confidence:
            lift = conf / item_support_dict[rhs]
            rules_info = lhs, rhs, conf, lift
            rules.append(rules_info)
            right_hand_side.append(rhs)

    return rules, right_hand_side


def create_rules(freq_items, item_support_dict, min_confidence):
    """
    create the association rules, the rules will be a list.
    each element is a tuple of size 4, containing rules'
    left hand side, right hand side, confidence and lift
    """
    association_rules = []
    for idx, freq_item in enumerate(freq_items[1:(len(freq_items) - 1)]):
        for freq_set in freq_item:

            subsets = [frozenset([item]) for item in freq_set]
            rules, right_hand_side = compute_conf(freq_items,
                                                  item_support_dict,
                                                  freq_set, subsets,
                                                  min_confidence)
            association_rules.extend(rules)
            if idx != 0:
                k = 0
                while len(right_hand_side[0]) < len(freq_set) - 1:
                    ck = create_candidate_k(right_hand_side, k=k)
                    rules, right_hand_side = compute_conf(freq_items,
                                                          item_support_dict,
                                                          freq_set, ck,
                                                          min_confidence)
                    association_rules.extend(rules)
                    k += 1

    return association_rules


freq_items, item_support_dict = apriori(X_data, min_support=0.3)
association_rules = create_rules(freq_items, item_support_dict, min_confidence=0.5)


print("------------------------------")
print("Frequent Itemsets:")
for itemset in freq_items:
    for item in itemset:
        print('\t', item, "\tSupport:", item_support_dict[item])

print("\nAssociation Rules:")
for antecedent, consequent, confidence, lift in association_rules:
    print('\t', antecedent, "=>", consequent, "\tConfidence:", round(confidence, 4), "\tLift:", round(lift, 4))
print("------------------------------")
