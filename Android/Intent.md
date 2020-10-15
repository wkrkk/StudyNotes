#### Intent

**一、显式Intent**

通过指定目标组件名称来启动组件，并且每次启动的组件只能有一个。通常用于启动应用程序内部组件。

```
Intent intent=new Intent(getApplicationContext(),BActivity.class);
startActivity(intent);
```

**二、隐式Intent**

不指定要启动的目标组件名称，而是指定Intent的Action、Data或Category等，通常用隐式Intent激活其他应用程序中的组件。在启动组件时，会去匹配AndroidManifest.xml相关组件的Intent-filter。

隐式启动Activity是指Android系统根据过滤规则自动去匹配对应的Intent。

```
//过滤规则
<intent-filter>
    <!--cn.edu.nnutc.ie.MYACTION：标志名，过滤时根据此匹配-->
   <action android:name="cn.edu.nnutc.ie.MYACTION"/>
   <category android:name="android.intent.category.DEFAULT"/>
</intent-filter>
```



**三、Intent传递数据**

①传单个数据

```
//A活动
Intent intent=new Intent(getApplicationContext(),BActivity.class);
intent.putExtra("content",value);//value可以是很多简单类型的数据
startActivity(intent);

//B活动
Intent intent=getIntent();
String s=intent.getStringExtra("content");
```

②传多个数据，使用Bundle

```
//A活动
Intent intent=new Intent(getApplicationContext(),BActivity.class);
Bundle bundle=new Bundle();
bundle.putInt("age",12);
bundle.putString("name","Tom");
intent.putExtras(bundle);
startActivity(intent);

//B活动
Intent intent=getIntent();
Bundle bundle=intent.getExtras();
int age=bundle.getInt("age");
String name=bundle.getString("name");
```

