# -*-  coding: utf-8 -*-

age = 20
if age >= 18:
    print('adult')
elif age >= 6:
    print('teenage')
else:
    print('kid')

names = ('Lisa', 'Mary', 'Henry')
for x in names:
    print(x)


x = [1, 2, 3]
y = (1, 2, 3)

def fun(*num):
    sum = 0
    for x in num:
        sum = sum + x
    return sum

sum = fun(1, 2, 3)
print(sum)

s = [1, 2, 3]
sum = fun(*s)
print(sum)
