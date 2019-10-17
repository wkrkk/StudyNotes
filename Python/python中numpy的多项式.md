#### python中numpy的多项式

<u>1.从已知根求解多项式</u>

`from numpy import *`

`root=[1,-1]`                                              #一个多项式的根为1，-1

`poly1d(poly(root))`                               #求解出该多项式

<u>2.roots求解多项式的根</u>

`roots(a)`

`array_equal(root,roots(a))`              #判断两个根是否相等

<u>3.原函数与导函数</u>

`polyder(a)`                                               #求导函数

`polyint(a)`                                               #求原函数

<u>4.多项式在某点处的值</u>

`polyval(a,5)`                                           #在x=5处的值

<u>5.加减乘除四则运算</u>

`poly1d([1, 1])`                                       #构建多项式，[]内从高到低次幂的系数

`a.coeffs`                                                    #返回多项式的系数                                

`a.order`                                                      #返回多项式的阶数

`polyadd(a, b)、polysub(a, b)、polymul(a, b)、polydiv(a, b)`

​                                                                      #两多项式加减乘除

`roots(polysub(a, b))`                           #两个多项式的交点，就是多项式相减之后的零点