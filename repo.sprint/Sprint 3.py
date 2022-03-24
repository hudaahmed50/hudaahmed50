#!/usr/bin/env python
# coding: utf-8

# In[5]:


#q1 itertools.product()
#https://www.hackerrank.com/challenges/itertools-product/problem?isFullScreen=false
# Enter your code here. Read input from STDIN. Print output to STDOUT
# itertools.product() in Python - Hacker Rank Solution START
from itertools import product
A = input().split()
A = list(map(int,A))
B = input().split()
B = list(map(int, B))
output = list(product(A,B))
for i in output:
    print(i, end = " ");


# In[1]:


#https://www.hackerrank.com/challenges/itertools-permutations/problem?isFullScreen=false
#q2   itertools.permutations(iterable[, r])
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import permutations
s , n = input().split()

for i in list(permutations(sorted(s),int(n))):
    print(''.join(i))


# In[6]:


#q3 itertools.combinations()
#https://www.hackerrank.com/challenges/itertools-combinations/problem?isFullScreen=false
# Enter your code here. Read input from STDIN. Print output to STDOUT
# itertools.combinations() in Python - Hacker Rank Solution START
from itertools import combinations

io = input().split()
S = io[0]
k = int(io[1])
for i in range(1,k+1):
    for j in combinations(sorted(S),i):
        print("".join(j))


# In[2]:


#Q4 itertools.combinations_with_replacement()
#https://www.hackerrank.com/challenges/itertools-combinations-with-replacement/problem?isFullScreen=false
from itertools import combinations_with_replacement

io = input().split();
char = sorted(io[0]);
N = int(io[1]);

for i in combinations_with_replacement(char,N):
    print(''.join(i));


# In[7]:


#q5 Compress the String!
#https://www.hackerrank.com/challenges/compress-the-string/problem?isFullScreen=false
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Compress the String in python - Hacker Rank Solution START
from itertools import *

io = input()
for i,j in groupby(map(int,list(io))):
    print(tuple([len(list(j)), i]) ,end = " ")


# In[3]:


#Q6 Iterables and Iterators
#https://www.hackerrank.com/challenges/iterables-and-iterators/problem?isFullScreen=false
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import combinations

N = int(input())
L = input().split()
K = int(input())

C = list(combinations(L, K))
F = filter(lambda c: 'a' in c, C)
print("{0:.3}".format(len(list(F))/len(C)))


# In[4]:


#Q7 Maximize It!
#https://www.hackerrank.com/challenges/maximize-it/problem?isFullScreen=false
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import product
K,M = map(int,input().split())
nums = []
for _ in range(K):
    row = map(int,input().split()[1:])
    nums.append(map(lambda x:x**2%M, row))
print(max(map(lambda x: sum(x)%M, product(*nums))))


# In[ ]:




