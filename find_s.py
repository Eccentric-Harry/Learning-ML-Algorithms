import csv

def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        traindata = []
        for row in datareader:
            traindata.append(row)
    return traindata

# Initial hypothesis
h = ['phi', 'phi', 'phi', 'phi', 'phi', 'phi']
data = read_data('./datasets/finds.csv')

def isConsistent(h, d):
    if len(h) != len(d) - 1:
        print("Number of attributes are not the same in hypothesis.")
        return False
    else:
        matched = 0
        for i in range(len(h)):
            if (h[i] == d[i]) or (h[i] == 'any'):
                matched += 1
        return matched == len(h)

def makeConsistent(h, d):
    for i in range(len(h)):
        if h[i] == 'phi':
            h[i] = d[i]
        elif h[i] != d[i]:
            h[i] = 'any'
    return h

print('Begin: Hypothesis:', h)

for d in data:
    if d[len(d) - 1] == 'Yes':
        if not isConsistent(h, d):
            h = makeConsistent(h, d)
        print("Training data:", d)
        print('Updated Hypothesis:', h)
        print()

print('Maximally specific data set End: Hypothesis:', h)
