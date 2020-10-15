#### SharedPreferences存储访问机制

Android系统中对于**数据量少、属于基本类型（int、String...）**的数据通常采用轻量级的存储类——SharedPreferences。

①存储数据

```
//获取SharePreferences对象,最终将name保存到系统中
SharedPreferences sharedPreferences=this.getSharedPreferences("login_content",MODE_PRIVATE);

//通过SharedPreferences对象的edit()方法获得SharedPreferences.Editor对象
SharedPreferences.Editor editor=sharedPreferences.edit();

//通过Editor对象的putXXX()方法，将不同类型的数据以key-value键值对的形式存储
editor.putString(键,值);
例：editor.putString("pwd",pwd);

//通过Editor对象的commit()方法提交数据，也就是将数据保存到xml文件中
editor.commit();
```

②读取数据

```
//获取SharedPreferences对象实例
SharedPreferences sharedPreferences=this.getSharedPreferences("login_content",MODE_PRIVATE);
//取出数据
String name=sharedPreferences.getString(键,默认值);
例：String pwd=sharedPreferences.getString("pwd","");
```

