#### Spinner下拉列表框

Spinner是ViewGroup的间接子类，可以作为容器使用。

Spinner提供了从一个数据集合中快速选择一项值的办法，默认情况下选择的是当前选择的值，点击Spinner会弹出一个包含所有可选值的dropdow菜单，从中选择一个新值。

**使用Spinner，必须给它准备数据，并且读数据是用ArrayAdapter装配的。**

```
//ArrayAdapter的三个参数：上下文，布局，数据（字符串数组）
ArrayAdapter adapter=new ArrayAdapter(getApplicationContext(),android.R.layout.simple_dropdown_item_1line, provinces);

//ArrayAdapter的三个参数：上下文，数据源（xml），布局
ArrayAdapter adapter=ArrayAdapter.createFromResource(getApplicationContext(),R.array.provinces_array,android.R.layout.simple_spinner_dropdown_item);

//将构建好的适配器对象传递进去，完成Spinner与数据的关联
spinner.setAdapter(adapter);
```

**ArrayAdapter**的三个参数：上下文，布局，数据

一、上下文

`getApplicationContext()`

二、布局

对于Spinner项的布局可以使用系统提供的，也可以使用用户自定义的

常使用`android.R.layout.simple_dropdown_item_1line`

三、数据

数据来源：字符串数组，xml

①使用数组

```
String[] provinces=new String[]{"江苏省","山西省","北京市","上海市","天津市"};
```

②在strings中设置

```
<string-array name="provinces_array">
        <item>江苏省</item>
        <item>山西省</item>
        <item>北京市</item>
        <item>上海市</item>
        <item>天津市</item>
    </string-array>
```

**下拉列表选择新值另一方式——在准备好xml数据后在Spinner控件中添加属性：android:entries="@array/provinces_array"**

