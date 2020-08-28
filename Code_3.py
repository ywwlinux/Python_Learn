# -*-  coding: utf-8 -*-

'Source code for learning oop - class'

__author__ = "Wenwu Yang"

class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    
    def print_score(self):
        print('%s: %s'%(self.name, self.score))

bart = Student('FangFang', 80)
bart.print_score()

