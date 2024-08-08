import numpy as np
import pandas as pd
import math
import copy

# Load the dataset
dataset = pd.read_csv('./datasets/Tennis.csv')  # Make sure 'tennis.csv' is in the same directory as this script
X = dataset.iloc[:, :].values

# Define the attributes (column names excluding the target)
attribute = ['Outlook', 'Temp', 'Humidity', 'Wind']

# Node class definition
class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = []

# Entropy calculation function
def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0

    for i in rows:
        if data[i][idx] == 'Yes':
            yes += 1
        else:
            no += 1

    x = yes / (yes + no)
    y = no / (yes + no)
    if x != 0 and y != 0:
        entropy = -1 * (x * math.log2(x) + y * math.log2(y))
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans

# Information gain calculation
def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        for i in rows:
            key = data[i][j]
            mydict[key] = mydict.get(key, 0) + 1

        gain = entropy
        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Yes':
                        yes += 1
                    else:
                        no += 1
            x = yes / (yes + no)
            y = no / (yes + no)
            if x != 0 and y != 0:
                gain += (mydict[key] * (x * math.log2(x) + y * math.log2(y))) / len(rows)

        if gain > maxGain:
            maxGain = gain
            retidx = j

    return maxGain, retidx, ans

# Build the decision tree
def buildTree(data, rows, columns):
    maxGain, idx, ans = findMaxGain(data, rows, columns)
    root = Node()
    root.childs = []

    if maxGain == 0:
        root.value = 'Yes' if ans == 1 else 'No'
        return root

    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        mydict[key] = mydict.get(key, 0) + 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = [i for i in rows if data[i][idx] == key]
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root

# Tree traversal function
def traverse(root):
    print(root.decision)
    print(root.value)

    if root.childs:
        for child in root.childs:
            traverse(child)

# Calculate and build the decision tree
def calculate():
    rows = [i for i in range(0, len(X))]
    columns = [i for i in range(0, len(attribute))]
    root = buildTree(X, rows, columns)
    root.decision = 'Start'
    traverse(root)

# Execute the decision tree calculation
calculate()
