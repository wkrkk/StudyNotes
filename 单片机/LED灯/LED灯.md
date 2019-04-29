##### 点亮第一个LED灯

```
#include <reg52.h>

sbit LED1 = P1^0;
sbit LED8 = P1^7;

void main()
{
	   LED1=0; //点亮led1灯
	   LED8=0;//同时点亮led1、led8
}
```

##### LED闪烁

```
#include<reg52.h>

unsigned int i;	  //0~65535

void main()
{
   while(1)
   {
		//P1=0;
		P1=0xe9;//1110 1001
		i=65535;
		while(i--); //用于延时
		P1=0xff;//1111 1111
		i=65535;
		while(i--);	
   }
  	
}
```

##### 流水灯

```
#include<reg52.h>
#include<intrins.h>//循环移位函数的标准库

//宏定义
#define uint unsigned int
#define uchar unsigned char

uchar temp;

//毫秒级延时函数定义
void delay(uint z)
{
	  uint x,y;
	  for(x=z;x>0;x--)
	     for(y=114;y>0;y--);
}
void main()
{
	temp=0xfe; //1111 1110 
	P1=temp;
	delay(100);//延时100毫秒
	while(1)
	{
		temp=_crol_(temp,1);//循环左移一位，_cror_循环右移
		P1=temp;
		delay(100);
	}	
}
```

<循环左移：最高位移到最低位  1111 1110 --->   1111 1101>

<左移运算符：最高位移除，最低为补0  1111 1110 --->   1111 1100>