import numpy as np


def main():
    filename = '/Users/wdaugherty/Cornell_Tech_DL/CS-5787/HW2/data/iris-test.txt'
    test_data = []
    for line in open(filename):
        item = line.split()
        test_data.append(item)
    test_data = np.array(test_data)
    print(test_data[:5])
    print(type(test_data))

# main()

t = [1, 2, 3, 4, 5]
print(t[:2])