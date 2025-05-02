from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Rule:
    id: int
    predicates: list
    class_label: int


    def as_string(self):
        s = "IF"
        for idx, pred in enumerate(self.predicates):
            f, op, th = pred
            s += " {} {} {:.4f}".format(f, op, th)
            if idx != len(self.predicates) - 1:
                s += " AND"
        s += " THEN PREDICT class = {}".format(self.class_label)
        return s
    
    def check_equal_preds(self, rule2):
        preds1 = self.predicates
        preds2 = rule2.predicates

        if len(preds1) != len(preds2):
            return False
        for pred1 in preds1:
            if pred1 not in preds2:
                return False

        return True

    def __repr__(self):
        return self.as_string()
    
    def as_dict(self,feature_names: list = None, total_region: dict = None, only_preds = False) -> dict:
        if feature_names == None:
            feature_names = []
            for pred in self.predicates:
                f, _, _ = pred
                feature_names.append(f)
        feature_names = list(set(feature_names))
        region = {}
        for feature in feature_names:
            if total_region is not None:
                lb = total_region[feature][0]
                ub = total_region[feature][1]
            else:
                lb = -np.inf
                ub = np.inf
            
            pred_having_feature = []
            for pred in self.predicates:
                f,op,th = pred
                if f == feature:
                    pred_having_feature.append(pred)
            
            for idx, pred in enumerate(pred_having_feature):
                f, op, th = pred
                if op == '<=' or op == '<':
                    if th < ub:
                        ub = th
                elif op == '>=' or op == '>':
                    if th > lb:
                        lb = th
            if not only_preds:
                region[feature] = [lb,ub]
            elif len(pred_having_feature) != 0:
                region[feature] = [lb,ub]
        return region

    def apply(self,X: pd.DataFrame):
        res = np.ones(X.shape[0],dtype='bool')
        for idx, pred in enumerate(self.predicates):
            f,op,th = pred
            if op == '<=' or op == '<':
                if idx == 0:
                    res = (X[f] <= th).to_numpy()
                else:
                    res = res*(X[f] <= th).to_numpy()

            elif op == '>=' or op == '>':
                if idx == 0:
                      res = (X[f] >= th).to_numpy()
                else:
                    res = res*(X[f] >= th).to_numpy()
            else:
                raise ValueError('op must be <= or < or >= or >')
        return res
    
    def filter(self,X:pd.DataFrame):
        return X[self.apply(X)]
    
    @staticmethod
    def intersect_dicts(region1: dict, region2: dict):
        result = {}
        features1 = list(region1.keys())
        features2 = list(region2.keys())
        all_features = list(set(features1).union(features2))

        for feature in all_features:
            if feature in region1.keys() and feature not in region2.keys():
                result[feature] = region1[feature]
            elif feature not in region1.keys() and feature in region2.keys(): 
                result[feature] = region2[feature]
            else:
                l1 = region1[feature][0]
                r1 = region1[feature][1]
                l2 = region2[feature][0]
                r2 = region2[feature][1]
                if l2>=r1 or l1>=r2:
                    return {}
                else:
                    result[feature] = [max(l1,l2),min(r1,r2)]
        return result
    
    def intersection(self,another_rule):
        region1 = self.as_dict()
        region2 = another_rule.as_dict()

        region = Rule.intersect_dicts(region1,region2)
        new_preds = []

        new_class_label = int(self.class_label != another_rule.class_label)

        for feature in region.keys():
            lb = region[feature][0]
            ub = region[feature][1]

            if not np.isinf(lb):
                new_preds.append((feature,'>',lb))
            if not np.isinf(ub):
                new_preds.append((feature,'<=',ub))
        
        new_id = str(self.id) + str(another_rule.id)
        return Rule(new_id,new_preds,new_class_label)

    



        




              

    
    