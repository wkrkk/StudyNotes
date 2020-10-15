#### RadioButton、CheckBox、RatingBar

###### RadioButton

RadioButton是button的子类，拥有所有Button的属性和方法。

RadioGroup类继承LinearLayout类（具有水平和垂直方向），一般作为RadioButton的容器，可以添加多个RadioButton；同一个RadioGroup中的RadioGroup相互排斥。

例如：

![](https://github.com/wkrkk/RandomPictures/blob/master/Android/RadioButton.png?raw=true)

###### CheckBox复选框

实现多个选项同时选中的功能，Button的子类。

例如:

![](https://github.com/wkrkk/RandomPictures/blob/master/Android/CheckBox.png?raw=true)

###### RatingBar评分条

一、默认风格

系统内置，直接使用控件

`style="?android:ratingBarStyle"`

![](https://github.com/wkrkk/RandomPictures/blob/master/Android/RatingBar.png?raw=true)

二、自定义风格

①在drawable下放置两张图片

②在drawable下添加xml，root:layer-list

```
<!--三个id是系统所给，不能修改-->
    <!--定义未选中图片作为背景-->
    <item
        android:id="@android:id/background"
        android:drawable="@drawable/heart_1">
    </item>

    <!--定义未选中图片作为第二进度-->
    <item
        android:id="@android:id/secondaryProgress"
        android:drawable="@drawable/heart_1">
    </item>

    <!--定义选中图片作为第一进度-->
    <item
        android:id="@android:id/progress"
        android:drawable="@drawable/heart_2">
    </item>
```

③在styles中自定义风格

```
<style name="RatingStyle" parent="@android:style/Widget.RatingBar">
        <!--定义星星图片样式，直接引用drawable下创建的xml-->
        <item name="android:progressDrawable">@drawable/myratingbar</item>
    </style>
```

④RatingBar控件中设置：`style="@style/RatingStyle"`