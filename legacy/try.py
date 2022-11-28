import numpy as np

if __name__ == '__main__':
    base = np.array([0.1 + x for x in range(0, 10)])
    schedule = np.array(0.11 + base)
    print(schedule)