import numpy as np
import pandas as pd

def H( y , mode='entropy'):
        if isinstance(y,pd.Series):
            y = y.to_numpy()
        if y.shape[0] == 0:
            return 0.0
        all_classes = np.unique(y)

        p = (y == all_classes[:,None]).sum(axis = 1) ### check this
        print(p)
        p = p / y.shape[0]

        if mode == 'entropy':
            epsilon = 1e-8
            return -(p * np.log2( p+ epsilon)).sum()
        elif mode == 'gini':
            return 1 - ((p**2).sum())
        

y = np.array([1,1,1,2,3,5,5])
print(H(y))