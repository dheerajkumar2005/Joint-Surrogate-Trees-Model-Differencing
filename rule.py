from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Rule:
    _id: int
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
    
    