import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from rule import Rule

def leaf_to_rule(leaf: dict) -> Rule : 
    """
    converts leaf to Rule
    Args:
        leaf: Dict with keys: path -> all predicates leading to leaf
                              val -> label of the node
    Return:
        Rule object
    """

    return Rule( 0, leaf['path'], leaf['val'])


def get_leaves(root: dict) -> list[dict] :
    """
    returns the list of leaf nodes in a tree
    Args:
        root:   A dictionary representing the root of a decision tree.
                Each node is a dict with keys like 'cutoff' (for split nodes),
                'left', 'right' (children), or 'val', 'path' (for leaves)
    """
    nodes = []
    def _recurse(root):
        is_split_node = ('cutoff' in root)
        if is_split_node:
            _recurse(root['left'])
            _recurse(root['right'])
        else:
            nodes.append(root)
    _recurse(root)
    return nodes

"""
Tree Structure:
For Split Node (internal node):
'col': The feature name used for the split (e.g., 'X[22]').
'index_col': The feature index (e.g., 22).
'cutoff': The threshold for the split (e.g., 116.05).
'val': The majority class label at this node (used if the node becomes a leaf).
'depth': The depth of the node in the tree.
'dist': The distribution of class labels (via np.bincount(y)).
'left': The left child node (for data where feature < cutoff).
'right': The right child node (for data where feature >= cutoff).

For Leaf Node:
'val': The predicted class label.
'depth': The depth of the leaf.
'ispure': A boolean indicating if the leaf is pure (all labels are the same).
'dist': The distribution of class labels.
'path': A list of predicates (conditions) from the root to this leaf, e.g., [('X[22]', '>=', 116.05), ('X[29]', '<', 0.1)].

"""

class JointSurrogateTree:
    
    def __init__(self, max_depth, feature_names, alpha = 0, split_criterion = 1):
        # self.depth1 = 0
        # self.depth2 = 0
        self.alpha = alpha
        self.max_depth = max_depth
        self.feature_names = feature_names

        if split_criterion == 1:
            self.continue_same_prefix = self.split_condition
        else:
            self.continue_same_prefix = self.split_condition2
        
        self.diff_rules = []

    def H(self, y , mode='entropy'):
        if isinstance(y,pd.Series):
            y = y.to_numpy()
        if y.shape[0] == 0:
            return 0.0
        all_classes = np.unique(y)

        p = (y == all_classes[:,None]).sum(axis = 1)
        p = p / y.shape[0]

        if mode == 'entropy':
            epsilon = 1e-8
            return -(p * np.log2( p+ epsilon)).sum()
        elif mode == 'gini':
            return 1 - ((p**2).sum())
        else:
            raise ValueError('Mode should be either gini or entropy')

    def split_condition(self,impurity1,impurity2,impurity):
        return ((impurity1 > 0) and (impurity2 > 0))
    
    def split_condition2(self,impurity1,impurity2,impurity):
        if impurity1 <= 0 or impurity2 <= 0 or impurity <= 0:
            return False
        im_avg = (impurity1 + impurity2) / 2
        return self.alpha*impurity < im_avg
    
    def get_impurity_split(self, y_pred, y_real, mode = 'entropy'):
        assert(y_pred.shape == y_real.shape)

        y_left = y_real[y_pred]
        y_right = y_real[~y_pred]

        n_l = len(y_left)
        n_r = len(y_right)

        n = n_l + n_r
        if n == 0:
            return 0.0
        p_l = n_l / n
        p_r = n_r / n

        return p_l*self.H(y_left,mode) + p_r*self.H(y_right,mode)

    def find_best_feature_to_split_for_st(self, x:np.ndarray, y:np.ndarray):
        if isinstance(x,pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y,pd.Series):
            y = y.to_numpy()

        col = None
        min_imp = 10
        cutoff = None

        for idx, c in enumerate(x.T):
            unique_values = np.unique(c)
            split_points = (unique_values[:-1] + unique_values[1:])/2
            for value in split_points:
                y_pred = c < value
                cur_imp = self.get_impurity_split(y_pred, y)

                if cur_imp == 0:
                    return idx, value, cur_imp
                
                elif cur_imp < min_imp:
                    min_imp = cur_imp
                    col = idx
                    cutoff = value

        return col, cutoff, min_imp
    
    def find_best_feature_to_split_for_jst(self, x1 : np.ndarray, y1: np.ndarray, x2 : np.ndarray, y2 : np.ndarray):
        if isinstance(x1, pd.DataFrame):
            x1 = x1.to_numpy()
        if isinstance(y1, pd.Series):
            y1 = y1.to_numpy()
        if isinstance(x2, pd.DataFrame):
            x2 = x2.to_numpy()
        if isinstance(y2, pd.Series):
            y2 = y2.to_numpy()
        
        col = None
        min_imp = 10
        cutoff = None

        for idx, c in enumerate(x1.T):
            unique_values = np.unique(c)
            split_points =  (unique_values[:-1] + unique_values[1:])/2

            for value in split_points:
                y_pred = c < value
                cur_imp1 = self.get_impurity_split(y_pred, y1)
                cur_imp2 = self.get_impurity_split(y_pred, y2)
                cur_imp = (cur_imp1 + cur_imp2)/2

                if cur_imp == 0:
                    return idx, value, cur_imp
                
                elif cur_imp < min_imp:
                    min_imp = cur_imp
                    col = idx
                    cutoff = value
        
        return col, cutoff , min_imp
    
    def is_split_pure(self, y: np.ndarray):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if len(y) == 0: return True
        else: return len(np.unique(y)) == 1
    
    def make_st(self, x, y , path=None, depth = 0):
        if path is None:
            path = []
        if len(y) == 0:
            return None
        elif self.is_split_pure(y):
            return {'val': y[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y), 'path': path }
        elif depth >= self.max_depth:
            label = np.argmax(np.bincount(y))
            return {'val': label, 'depth': depth, 'ispure': self.is_split_pure(y), 'dist': np.bincount(y), 'path': path}
        else:
            col, cutoff, min_imp = self.find_best_feature_to_split_for_st(x,y)
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:,col] >= cutoff]
            x_left = x[x[:, col] < cutoff]
            x_right = x[x[:,col] >= cutoff]
            left_path = path + [(self.feature_names[col], '<', np.round(cutoff, 4))]
            right_path = path + [(self.feature_names[col], '>=', np.round(cutoff, 4))]
            
            split_node = {
                          'col': self.feature_names[col], 'index_col': col, 'cutoff': cutoff,
                          'val': np.argmax(np.bincount(y)), 'depth': depth, 'dist': np.bincount(y),
                          'left': self.make_st( x_left,y_left, left_path, depth+1),
                          'right': self.make_st( x_right,y_right, right_path, depth+1)
                        }
            return split_node
    
    def make_jst(self,x1,y1,x2,y2, path=None, depth =0):
        if path is None:
            path = []
        if len(y1) == 0 and len(y2) == 0:
            return None,None
        if len(y1) == 0:
            return None, self.make_st(x2,y2,path,depth)
        elif len(y2) == 0:
            return self.make_st(x1,y1,path,depth),None
        
        if self.is_split_pure(y1) and self.is_split_pure(y2):
            
            return {'val': y1[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y1), 'path': path},\
                   {'val': y2[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y2), 'path': path }

        if self.is_split_pure(y1):
            return {'val': y1[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y1), 'path': path },\
                   self.make_st(x2,y2,path,depth)
        
        elif self.is_split_pure(y2):
            return self.make_st(x1,y1,path,depth),\
                   {'val': y2[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y2), 'path': path }

        if depth >= self.max_depth:
            label1 = np.argmax(np.bincount(y1))
            label2 = np.argmax(np.bincount(y2))
            return {'val': label1, 'depth': depth, 'ispure': self.is_split_pure(y1), 'dist': np.bincount(y1), 'path': path},\
                   {'val': label2, 'depth': depth, 'ispure': self.is_split_pure(y2), 'dist': np.bincount(y2), 'path': path}
        
        col1, cutoff1, impurity1 = self.find_best_feature_to_split_for_st(x1,y1)
        col2, cutoff2, impurity2 = self.find_best_feature_to_split_for_st(x2,y2)
        col, cutoff, impurity = self.find_best_feature_to_split_for_jst(x1,y1,x2,y2)
        if self.continue_same_prefix(impurity1, impurity2, impurity):            
            split_node_1 = {
                            'col': self.feature_names[col], 'index_col': col, 'cutoff': cutoff,
                            'val': np.argmax(np.bincount(y1)), 'depth': depth, 'dist': np.bincount(y1)
                            }
            split_node_2 = {
                            'col': self.feature_names[col], 'index_col': col, 'cutoff': cutoff,
                            'val': np.argmax(np.bincount(y2)), 'depth': depth, 'dist': np.bincount(y2)
                            }
            y1_left = y1[x1[:,col] < cutoff]
            y1_right = y1[x1[:,col] >= cutoff]

            x1_left = x1[x1[:,col] < cutoff]
            x1_right = x1[x1[:,col] >= cutoff]

            y2_left = y2[x2[:, col] < cutoff]
            y2_right = y2[x2[:, col] >= cutoff]

            x2_left = x2[x2[:,col] < cutoff]
            x2_right = x2[x2[:,col] >= cutoff]

            left_path = path + [(self.feature_names[col], '<', np.round(cutoff,4))]
            right_path = path + [(self.feature_names[col], '>=', np.round(cutoff,4))]

            split_node_1['left'], split_node_2['left']  = self.make_jst(x1_left,y1_left,x2_left,y2_left,left_path,depth+1)
            split_node_1['right'], split_node_2['right']  = self.make_jst(x1_right,y1_right,x2_right,y2_right,right_path,depth+1)


        else:
            y1_left = y1[x1[:,col1] < cutoff1]
            y1_right = y1[x1[:,col1] >= cutoff1]

            x1_left = x1[x1[:,col1] < cutoff1]
            x1_right = x1[x1[:,col1] >= cutoff1]

            left_path1 = path + [(self.feature_names[col1], '<', np.round(cutoff1,4))]
            right_path1 = path + [(self.feature_names[col1], '>=', np.round(cutoff1,4))]

            split_node_1 = {
                          'col': self.feature_names[col1], 'index_col': col1, 'cutoff': cutoff1,
                          'val': np.argmax(np.bincount(y1)), 'depth': depth, 'dist': np.bincount(y1),
                          'left': self.make_st( x1_left,y1_left, left_path1, depth+1),
                          'right': self.make_st( x1_right,y1_right, right_path1, depth+1)
                        }
            y2_left = y2[x2[:,col2] < cutoff2]
            y2_right = y2[x2[:,col2] >= cutoff2]

            x2_left = x2[x2[:,col2] < cutoff2]
            x2_right = x2[x2[:,col2] >= cutoff2]

            left_path2 = path + [(self.feature_names[col2], '<', np.round(cutoff2,4))]
            right_path2 = path + [(self.feature_names[col2], '>=', np.round(cutoff2,4))]

            split_node_2 = {
                          'col': self.feature_names[col2], 'index_col': col2, 'cutoff': cutoff2,
                          'val': np.argmax(np.bincount(y2)), 'depth': depth, 'dist': np.bincount(y2),
                          'left': self.make_st( x2_left,y2_left, left_path2, depth+1),
                          'right': self.make_st( x2_right,y2_right, right_path2, depth+1)
                        }
            
        return split_node_1, split_node_2
    
    def get_prediction(self, x, tree:dict):
        cur_layer = tree
        while cur_layer and ('cutoff' in cur_layer):
            if x[cur_layer['index_col']] < cur_layer['cutoff']:
                if cur_layer['left'] is None:
                    return cur_layer['val']
                cur_layer = cur_layer['left']
            else:
                if cur_layer['right'] is None:
                    return cur_layer['val']
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val',default=-1)

    def predict(self, x:np.ndarray, tree:dict):
        results = np.zeros((x.shape[0],))
        for i,row in enumerate(x):
            results[i] = self.get_prediction(row,tree)
        return results
    
    def accuracy(self, x, y_true, tree):
        y_pred = self.predict(x,tree)
        if np.any(y_pred == -1):
            raise ValueError("Some predictions are invalid (None returned by get_prediction)")
        acc = accuracy_score(y_true, y_pred)
        return np.round(acc,4)*100

    def common_trunk(self,t1:dict,t2:dict):
        t = {}
        keys1 = t1.keys()
        keys2 = t2.keys()

        same_keys = keys1 == keys2
        is_split_node = 'cutoff' in t1

        if same_keys and is_split_node:
            same_split_cond = (t1['col'] == t2['col']) and (t1['cutoff'] == t2['cutoff'])
        else:
            same_split_cond = False

        if same_keys and is_split_node and same_split_cond:
            for key in t1:
                if key not in ['left','right']:
                    t[key] = t1[key]
            t['left'] = self.common_trunk(t1['left'],t2['left'])
            t['right'] = self.common_trunk(t1['right'], t2['right'])
        else:
            t['tree1'] = t1
            t['tree2'] = t2

        return t
    
    def get_diffrules(self, root1: dict, root2: dict):
        leaves1 = get_leaves(root1)
        leaves2 = get_leaves(root2)

        diffrules = []
        for leaf1 in leaves1:
            for leaf2 in leaves2:
                if leaf1['val'] != leaf2['val']:
                    rule1 = leaf_to_rule(leaf1)
                    rule2 = leaf_to_rule(leaf2)
                    intersection = rule1.intersection(rule2)

                    if len(intersection.predicates) > 0:
                        diffrules.append(intersection)
        return diffrules


    def get_diffrules_from_jst(self, root: dict):
        diffrules = []

        def _recurse(root : dict):
            is_diverging = 'tree1' in root
            
            if is_diverging:
                tree1 = root['tree1']
                tree2 = root['tree2']

                diffrules.extend(self.get_diffrules(tree1,tree2))
            
            else:
                if 'cutoff' in root:
                    _recurse(root['left'])
                    _recurse(root['right'])
        
        _recurse(root)
        return diffrules



