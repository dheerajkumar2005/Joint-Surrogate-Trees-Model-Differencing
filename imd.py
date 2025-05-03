import numpy as np
import pandas as pd
from jst import JointSurrogateTree
from rule import Rule

class IMDExplainer:
    
    def __init__(self):
        self.jst = None
        self.diffrules = []
        self.diff_regions = []
        self.feature_names = []
        

    def fit(self, X_train: pd.DataFrame, Y1, Y2, max_depth = 6, split_criterion = 1, alpha = 0.25, verbose = True):
        self.feature_names = X_train.columns.to_list()

        x1 = x2 = X_train.to_numpy()

        if not isinstance(Y1, np.ndarray):
            Y1 = Y1.to_numpy()
        if not isinstance(Y2, np.ndarray):
            Y2 = Y2.to_numpy()

        y1 = Y1
        y2 = Y2

        ydiff = (y1 != y2).astype(int)

        
        if verbose:
            print(f"diffs in X_train = {ydiff.sum()} / {len(ydiff)} = {(ydiff.sum() / len(ydiff) * 100):.2f}%")
        
        
        jst = JointSurrogateTree(max_depth,self.feature_names,alpha,split_criterion)
        
        t1, t2 = jst.make_jst(x1,y1,x2,y2)
        
        ct = jst.common_trunk(t1,t2)
        diff_rules = jst.get_diffrules_from_jst(ct)
        
        self.jst = ct
        self.diff_rules = diff_rules

        minimums = X_train.to_numpy().min(axis=0)
        maximums = X_train.to_numpy().max(axis=0)

        feature_ranges = [[minimums[i],maximums[i]] for i in range(len(minimums))]

        total_region_dict = {self.feature_names[i]: feature_ranges[i] for i in range(len(self.feature_names))}

        diff_rule_dict = dict(enumerate(self.diff_rules))
        self.diff_regions = [rule.as_dict(self.feature_names, total_region_dict, False) for _,rule in diff_rule_dict.items()]

    def in_region(self,regions,data:np.ndarray):
        res = np.zeros(data.shape[0])

        for region in regions:
            res1 = np.ones(data.shape[0])
            for feature in region.keys():
                f_index = self.feature_names.index(feature)
                col = data[:, f_index]
                left = region[feature][0]
                right = region[feature][1]
                res1 = np.logical_and(res1, np.logical_and(col >= left, col <= right))
            
            res = np.logical_or(res,res1)
        
        return res
    
    def explain(self):
        return self.diff_rules
    
    def metrics(self, x_test: pd.DataFrame, y_test1, y_test2, name='test'):

        if self.jst is None:
            print("jst not fitted yet, please call .fit method first!")
            return {}
        
        metrics = {}
        diff_samples = y_test1 != y_test2

        total_no_diff_samples = np.sum(diff_samples).astype(int)

        metrics['diffs'] = total_no_diff_samples
        metrics['samples'] = len(x_test)

        in_region_diff = self.in_region(self.diff_regions,x_test[diff_samples].to_numpy())
        diff_samples_in_region = np.sum(in_region_diff).astype(int)

        in_region_all = self.in_region(self.diff_regions,x_test.to_numpy())
        samples_in_region = np.sum(in_region_all).astype(int)

        metrics[name + "-precision"] = np.round(diff_samples_in_region / samples_in_region, 6)
        metrics[name + "-recall"] = np.round(diff_samples_in_region / total_no_diff_samples,6)
        metrics["num-rules"] = len(self.diff_regions)

        preds = []
        for rule in self.diff_rules:
            preds.extend(rule.predicates)
        preds = set(preds)
        metrics["num-unique-predicates"] = len(preds)
        return metrics






