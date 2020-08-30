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

def divideByzero():
    return 5/0

import logging
logging.basicConfig(level=logging.ERROR)

def main():
    try:
        divideByzero()
    except Exception as e:
        logging.exception(e)
    finally:
        print('Final Done')

fpath = 'E:/Git_Repository/Python_Learn/tests.txt'

#with open(fpath, 'r') as f:
#    s = f.read()
#    print(s)
        

from io import StringIO

s = StringIO()
s.write('hello, world')

import os

os.uname()