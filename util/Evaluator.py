#-*- coding: utf8 -*-
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as prfs

from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)

class Evaluator(): # 三分类
    def __init__(self, type_map: dict={}) -> None:
        self.reset()
        self.type_map = type_map
    
    def reset(self, ):
        self.ground_truth = []
        self.predictions = []
        self.right = []
    
    def update(self, ground_truth, predictions):
        self.ground_truth.extend(ground_truth)
        self.predictions.extend(predictions)
        self.right.extend(list(filter(lambda item: item[0] == item[1], zip(ground_truth, predictions))))
    
    def rprint(self, results: dict={}):
        for type_, data in sorted(results.items()):
            print(f'\t===== {type_} results ======\n\t', end='')
            for key, val in data.items():
                print(f'{key}: {val} ', end='')
            print()
    
    def compute(self, origins, predictions, founds):
        p = founds/predictions if predictions != 0 else 0
        r = founds/origins if origins != 0 else 0
        f1 = 2*p*r/(p+r) if p+r != 0 else 0
        return p, r, f1

    def result(self, ):
        print("Confusion Matrix :")
        print(confusion_matrix(self.ground_truth, self.predictions))
        print("Classification Report :")
        print(classification_report(self.ground_truth, self.predictions, digits=4))
        origins_counter = Counter(self.ground_truth)
        predictions_counter = Counter(self.predictions)
        right_counter = Counter([x[0] for x in self.right])
        results = {}
        for type_, origins in origins_counter.items():
            prediction_ = predictions_counter[type_]
            right_ = right_counter[type_]
            p,r,f1 = self.compute(origins, prediction_, right_)
            results.setdefault(self.type_map[type_], {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)})
        origins = len(self.ground_truth)
        predictions = len(self.predictions)
        founds = len(self.right)
        p,r,f1 = self.compute(origins, predictions, founds)
        results.setdefault('Macro F1', {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4)})
        prfs_score = prfs(self.ground_truth, self.predictions, average='weighted')
        results.setdefault('Weighted F1', {'precision': round(prfs_score[0], 4), 'recall': round(prfs_score[1], 4), 'f1': round(prfs_score[2], 4)})
        self.rprint(results)
        evaluate_value = (results[self.type_map[2]]['f1']*3 + results['Weighted F1']['f1']*2 + results['Macro F1']['f1'])/6
        return results, evaluate_value
        
        