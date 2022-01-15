#-*- coding: utf8 -*-
class Average():
    def __init__(self) -> None:
        self.reset()
    
    def update(self, loss):
        self.loss += loss
        self.count += 1
    
    def average(self, ):
        return self.loss/self.count
    
    def reset(self):
        self.loss = 0
        self.count = 0

class EvaluatorAverger():
    def __init__(self, types) -> None:
        self.types = types
        self.reset(self.types)
    
    def reset(self, types):
        self.results = {}
        self.count = 0
        for t in types:
            self.results.setdefault(t, {'precision': 0, 'recall': 0, 'f1': 0})
        
    def update(self, temp_results):
        for t in temp_results:
            self.results[t]['precision'] += temp_results[t]['precision']
            self.results[t]['recall'] += temp_results[t]['recall']
            self.results[t]['f1'] += temp_results[t]['f1']
        self.count += 1
    
    def rprint(self, results: dict={}):
        for type_, data in sorted(results.items()):
            print(f'\t===== {type_} results ======\n\t', end='')
            for key, val in data.items():
                print(f'{key}: {val} ', end='')
            print()

    def average(self):
        for t in self.results:
            self.results[t]['precision'] /= self.count
            self.results[t]['recall'] /= self.count
            self.results[t]['f1'] /= self.count
        self.rprint(self.results)
        return self.results
