# Python序列

## 序列结构

|       对象类型        |             示例             |                           简要说明                           |
| :-------------------: | :--------------------------: | :----------------------------------------------------------: |
|      列表(list)       |  [1,2,3], ['a','b',['c',2]]  | 有序可变序列，所有元素放入[]中，元素之间逗号隔开，可以为任意类型 |
|      元组(tuple)      |       (2,-5,6),  (3,)        |      不可变序列，元组中只有一个元素时，后面逗号不能省略      |
|      字典(dict)       |     {1:'food',2:'taste'}     |               无序可变序列，元素形式为“键：值”               |
| 集合 (set、frozenset) |        {'a','b','c'}         | 只能包含数字、字符串等不可变类型，元素不能重复，set可变序列，frozenset不可变序列 |
|        字符串         | 'abc',  "124",  '''python''' |                 不可变序列，空串用''或""表示                 |

*列表、元素、字符串支持双向索引*

## 列表

#### 列表创建与删除

##### 创建

1.“=”直接将列表赋值给变量

`>>>a_list=['a','b','m']`

`>>>a_list=[]`

2.使用list()函数将元组、range对象、字符串或其他类型的可迭代对象类型的数据转换为列表

`>>>a_list=list((3,5,7))`

`>>>a_list=list(range(1,10,2))`

##### 删除

`del a_list`

#### 列表元素添加

1.“+”将元素添加到列表中

实质是创建了一个新列表将原列表和新加的元素添加进去

2.append()原地操作，在列表尾部进行添加

3.extend()原地操作，将另一个迭代对象的所有元素添加到该列表对象尾部

4.insert()将元素添加到列表的指定位置

`>>>aList.insert(3,6)`  #将下标为3的位置插入元素6

5."*"扩展列表对象

（1）[3,5,7]

​         ` >>>a_list=a_list*3`

​         [3,5,7,3,5,7,3,5,7]

（2）`>>>x=[[1,2,3]]*3`

​		  `>>>x[0][0]=10`

​		  [[10,2,3],[10,2,3],[10,2,3]]

#### 列表元素的删除

1.del删除列表中指定位置上的元素

`>>>del a_list[1]` 	#删除列表下标为1的元素

2.pop()删除并返回指定位置上的元素

`>>>a_list.pop()`	   #返回最后一个元素

`>>>a_list.pop(1)`	 #返回指定位置元素

3.remove()删除首次出现的指定元素

`>>>a_list.remove(7)`

#### 列表元素访问与计数

index()获取指定元素首次出现的下标

`>>>a_list.index(7)`

count()统计指定元素在列表对象中出现的次数

`>>>a_list.count(7)`

#### 成员资格判断

"in"判断一个值是否存在于列表中，返回结果为true或false

#### 切片操作

`>>>a_list[切片开始位置(0):切片截止位置(len):切片步长(1)]`

<u>浅复制</u>：生成一个新的列表，将原列表中所有元素的引用都复制到新列表

`>>>b_list=a_list[::]`		#切片，浅复制，两个列表的元素完全一样，但不是用一个对象，内存地址不同，修改一个不会影响另一个

#### 列表排序

1.sort()原地排序

`>>>a_list.sort()`								 #默认升序

`>>>a_list.sort(reverse=True)`		#降序

2.sorted()排序并返回新列表

`>>>sorted(a_list)`

`>>>sorted(a_list,reverse=True)`

3.reverse()原地逆序

`>>>a_list.reverse()`

4.reversed()逆序排列并返回迭代对象

`new_list=reversed(a_list)`

#### 用于序列操作的内置函数

1.len()、max()、min()、sum()

2.zip()返回可迭代的对象

```
>>>a_list=[1,2,3]
>>>b_liat=[4,5,6]
>>>c_liat=zip(a,b)
>>>list(c_list)			#把zip对象转换成列表
[(1,4),(2,5),(3,6)]
```

3.enumerate()枚举列表对象，返回的每个元素为包含下标和值的元组

```
>>>for item in enumerate('abcdef')
print(item)

(0,'a')
(1,'b')
...
```

#### 列表推导式

1.三种形式

（1）`>>>a_list=[x*x for x in range(10)]`

（2）

```
>>>a_list=[]
>>>for x in range(10):
	a_list.append(x*x)
```

（3）`>>>a_list=list(map(lambda x:x*x,range(10)))`

2.嵌套列表的平铺

（1）

```
>>>vec=[[1,2,3],[4,5,6],[7,8,9]]
>>>[num for elem in vec for num in elem]
[1,2,3,4,5,6,7,8,9]
```

（2）

```
>>>vec=[[1,2,3],[4,5,6],[7,8,9]]
>>>result=[]
>>>for elem in vec:
	for num in elem:
		result.append(num)
>>>result
[1,2,3,4,5,6,7,8,9]
```

3.过滤不符合条件的元素

```
>>>a_list=[-1,-4,6,7.5,-2.3,9,-11]
>>>[i for i in a_list if i>0]
[6,7.5,9]
```

4.多序列元素的任意组合

```
>>> [(x, y) for x in range(3) for y in range(3)]
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

>>> [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
```

5.矩阵转置

（1）

```
>>>matrix = [ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] 
>>> [[row[i] for row in matrix] for i in range(4)] 

[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]] 
```

（2）`>>>list(zip(*matrix))`  	#序列解包

6.使用函数或复杂表达式

（1）

```
>>> def f(v):
    if v%2 == 0:
        v = v**2
    else:
        v = v+1
    return v

>>> [f(v) for v in [2, 3, 4, -1] if v>0]

[4, 4, 16]
```

（2）`>>>[v**2 if v%2 == 0 else v+1 for v in [2, 3, 4, -1] if v>0]`

7.支持文件对象迭代

#### 列表实现向量运算

```
>>> import random
>>> x = [random.randint(1,100) for i in range(10)]            #生成随机数
>>> list(map(lambda i: i+5, x))                               #所有元素同时加5

>>> x = [random.randint(1,10) for i in range(10)]
>>> y = [random.randint(1,10) for i in range(10)]

>>> import operator
>>> sum(map(operator.mul, x, y))                     #向量内积
>>> sum((i*j for i, j in zip(x, y)))                 #向量内积
>>> list(map(operator.add, x, y))                    #两个等长的向量对应元素相加
```

## 元组

#### 元组创建与删除

##### 创建

1."="将一个元组赋值给变量

`>>>a=(3,)或3，`

`>>>a=()`

2.tuple将其他序列转换为元组

`>>>tuple('abcd')`

##### 删除

del删除元组对象，不能删除元组中的元素

## 字典

#### 字典创建与删除

##### 创建

1.“=”将一个字典赋值给一个变量

2.

（1）使用dict利用已有数据创建字典

```
>>> keys = ['a', 'b', 'c', 'd']
>>> values = [1, 2, 3, 4]
>>> dictionary = dict(zip(keys, values))

>>> dictionary
{'a': 1, 'c': 3, 'b': 2, 'd': 4}
```

（2）使用dict根据给定的键、值创建字典

```
>>> d = dict(name='Dong', age=37)

>>> d
{'age': 37, 'name': 'Dong'}
```

3.以给定内容为键，创建值为空的字典

```
>>> adict = dict.fromkeys(['name', 'age', 'sex'])

>>> adict
{'name': None,'age': None,'sex': None}
```

##### 删除

del删除整个字典

#### 字典元素的读取

1.以键作为下标读取字典元素，键不存在抛出异常

```
>>> aDict = {'name':'Dong', 'sex':'male', 'age':37}

>>> aDict['name']
'Dong'
```

2.get方法获取指定键对应的值，并且可以在键不存在的时候返回指定值

`>>>print(aDict.get('address'))`

​	None

`>>>print(aDict.get('address','SDIBT'))`

​	SDIBT

3.items()返回字典的键、值对列表

```
>>> aDict={'name':'Dong', 'sex':'male', 'age':37}
>>> for item in aDict.items():                   #输出字典中所有元素
	   print(item)
('age', 37)
('name', 'Dong')
('sex', 'male')

>>> for key in aDict:                            #不加特殊说明，默认输出键
	print(key)
age
name
sex

>>> for key, value in aDict.items():             #序列解包用法
	   print(key, value)
age 37
name Dong
sex male
```

4.keys()返回键列表

```
>>> aDict.keys()                                          #返回所有键
dict_keys(['name', 'sex', 'age'])
```

5.values()返回值列表

```
>>> aDict.values()                                       #返回所有值
dict_values(['Dong', 'male', 37])
```

6.当以指定键为下标为字典赋值时，键存在则可以修改该键的值，不存在则添加一个键、值对

`>>>aDict['age']=38`

7.update方法将另一个字典的键、值对添加到当前字典对象

`>>>aDict.update({'a':'a','b':'b'})`

#### 字典元素的添加与修改

1.del()删除指定键元素

2.clear()删除所有元素

3.pop()删除并返回指定键的元素

4.popitem()删除并返回字典中的一个元素

#### 有序字典

`>>>x=collections.OrderedDict()` 		#有序字典

## 集合

#### 集合的创建与删除

##### 创建

1.“=”

2.使用set将其他类型数据转换为集合

`>>>a_set=set(range(8,14))`

`>>>a_set=set([0,1,2,3,0,1,2])`			#自动去除重复

`>>>a_set=set()`										   #空集合

##### 删除

1.del()删除整个集合

2.pop()弹出并删除其中一个元素

3.remove()直接删除指定元素

4.clear()请空集合

#### 集合操作

集合支持交集、并集、差集等运算

#### 集合推导式

`>>>s={x.strip() for x in ('he','she','i')}`

​	{'i','she','he'}

## 序列解包

1.对多个变量同时赋值

`>>>x,y,z=1,2,3`					 		#多个变量同时赋值

`>>>x,y,z=range(3)`			  		#可以对迭代器对象进行序列解包

`>>>x,y,z=iter([1,2,3])`		   #使用迭代器对象进行解包

`>>>x,y,z=map(str,range(3))`   #使用可迭代map对象进行解包

`>>>a,b=b,a`									#交换两变量的值

`>>>x,y,z=sorted([1,3,2])`       #sorted返回排序后的列表

`>>>a,b,c='ABC'`							#字符串解包

2.字典

```
>>>s={'a':1,'b':2,'c':3}
>>>b,c,d=s.items()
>>>b,c,d
('a', 1)   ('b', 2)   ('c', 3) 

>>>b,c,d=s
>>>b,c,d
a   b   c

>>>b,c,d=s.values()
>>>b,c,d
1   2   3
```

3.序列

```
>>> keys = ['a', 'b', 'c', 'd']
>>> values = [1, 2, 3, 4]
>>> for k, v in zip(keys, values):
	  print((k, v), end=' ')
('a',1)
('b',2)
...
```

4.遍历enumerate对象（枚举列表对象）

```
>>> x = ['a', 'b', 'c']
>>> for i, v in enumerate(x):
	  print('The value on position {0} is {1}'.format(i,v))
	  
The value on position 0 is a
The value on position 1 is b
The value on position 2 is c
```

```
>>> aList = [1,2,3]
>>> bList = [4,5,6]
>>> cList = [7,8,9]
>>> dList = zip(aList, bList, cList)
>>> for index, value in enumerate(dList):
	    print(index, ':', value)

0 : (1, 4, 7)
1 : (2, 5, 8)
2 : (3, 6, 9)
```

## 生成器推导式

*生成器推导式的结果是一个生成器对象，使用生成器对象的元素时，可以根据需要转化为列表或元组，也可以用生成器对象__next__()方法和内置函数next()进行遍历，也可以直接作为迭代器对象使用*

***访问生成器对象，无法访问已访问过的元素***

1.__next__()或内置函数next()

`>>>g=((i+2)**2 for i in range(10))`			#创建生成器对象

`>>>tuple(g)`															#将其转换为元组

​    (4,9,16,25...)

`>>>list(g)`															  #生成器对象已遍历结束，没有元素了

​	[]

`>>>g._next_()`

​	4

`>>>next(g)`

​	9

2.使用for循环直接迭代生成器对象中的元素

```
>>> g = ((i+2)**2 for i in range(10))
>>> for item in g:                            #使用循环直接遍历生成器对象中的元素
       print(item, end=' ')
       
4 9 16 25 36 49 64 81 100 121 
```

```
>>> x = filter(None, range(20))       #filter对象也具有类似的特点

>>> 5 in x
True

>>> 2 in x                            #不可再次访问已访问过的元素
False

>>> 6 in x							  #6未被访问
True
```

```
>>> x = map(str, range(20))           #map对象也具有类似的特点

>>> '0' in x
True

>>> '0' in x                          #不可再次访问已访问过的元素
False
```

