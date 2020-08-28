# -*- coding: utf-8 -*-

'A testing standard document'

__author__ = 'Wenwu Yang'

import sys

def Test():
    args = sys.argv
    if (len(args)==1):
        print('No argument')
    elif (len(args)==2):
        print('Argument: ', args[1])
    else:
        print("More than 1 arguments")

if (__name__ == '__main__'):
    Test()