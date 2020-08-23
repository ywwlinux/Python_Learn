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

L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']

# Iterate on iterable objects
d = {'Henry':20, 'Mary':18, 'Kate':35}
for key in d:
    print(key)

for value in d.values():
    print(value)

for kv in d.items():
    print(kv)

s = 'I am hello'
for key in s:
    print(key)

for pair in enumerate(['A', 'B', 'C']):
    print(pair)



def createCounter():
    i = 0
    i = i+1
    return i

counterA = createCounter()
#print(counterA(), counterA(), counterA(), counterA(), counterA()) # 1 2 3 4 5