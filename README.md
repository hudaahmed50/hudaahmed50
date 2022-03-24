# HackerRank Sprint3 Task4



## 1- What ’re the methods that you used ?

 Python Regular Expressions
 Lambda function 
 filter() method 
 Python Lambda Functions
 re.split()
 group() 
 reduce()
 re.finditer()






## 2- Explain each method ..

### The re.search() method takes a regular expression pattern and a string and searches for that pattern within the string. If the search is successful,
 search() returns a match object or None otherwise. Therefore,
 the search is usually immediately followed by an if-statement to test if the search succeeded, as shown in the following example which searches for the pattern 'word:' followed by a 3 letter word (details below):

### filter(function, sequence)

function: function that tests if each element of a 
sequence true or not.

### re.split
   In re.split(), specify the regular expression pattern in the first parameter and the target character string in the second parameter.




### lambda arguments: 
expression
This function can have any number of arguments but only one expression, which is evaluated and returned.
One is free to use lambda functions wherever function objects are required.
You need to keep in your knowledge that lambda functions are syntactically restricted to a single expression.
It has various uses in particular fields of programming besides other types of expressions in functions.
re.split() 

 ###group([group1, ...])
Returns one or more subgroups of the match. If there is a single argument,
 the result is a single string; if there are multiple arguments, the result is a tuple with one item per argument.
 Without arguments, group1 defaults to zero (the whole match is returned). 
If a groupN argument is zero, the corresponding return value is the entire matching string;
 if it is in the inclusive range [1..99], it is the string matching the corresponding parenthesized group.
 If a group number is negative or larger than the number of groups defined in the pattern, an IndexError exception is raised.
If a group is contained in a part of the pattern that did not match, the corresponding result is None. 
If a group is contained in a part of the pattern that matched multiple times,
 the last match is returned.

### The reduce() function: 
applies a function of two arguments cumulatively on a list of objects in succession from left to right to reduce it to one value

### re.finditer()
The expression re.finditer() returns an iterator yielding MatchObject instances over all non-overlapping matches for the re pattern in the string.

### start([group])end([group])

Return the indices of the start and end of the substring matched by group;
 group defaults to zero (meaning the whole matched substring). 
Return -1 if group exists but did not contribute to the match. 
For a match object m, and a group g that did contribute to the match, the substring matched by group
## 3- What’s new for you ?
used Python Regular Expressions
filter() method
 Difference Between Lambda functions and def defined function
 re.finditer,
    



## 4- Resources ? 
https://developers.google.com/edu/python/regular-expressions
https://www.geeksforgeeks.org/filter-in-python/
https://www.geeksforgeeks.org/python-lambda-anonymous-functions-filter-map-reduce/
https://www.geeksforgeeks.org/gfact-50-python-end-parameter-in-print/
