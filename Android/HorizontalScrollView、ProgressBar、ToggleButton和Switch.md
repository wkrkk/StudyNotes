#### HorizontalScrollView、ProgressBar、ToggleButton和Switch

**HorizontalScrollView（水平滚动视图）：**用于布局的容器，可以放置多个视图。多个组件不能直接放置在滚动布局中，需放置在LinearLayout中。

**ProgressBar常用属性：**

|           属性            |              说明              |
| :-----------------------: | :----------------------------: |
|        android:max        |       设置进度条的最大值       |
|     android:progress      |       设置当前第一进度值       |
| android:secondaryProgress | 设置当前第二进度值设置是否显示 |
|           style           |     设置ProgressBar的风格      |

**ToggleButton、Switch：**继承CompoundButton类，用于两种状态改变时要实现的功能。例如：打开、关闭WIFI。