#### TabHost、TabWidget常用属性、方法

###### TabHost

**TabHost**继承自FrameLayout，是一个带Tab选项卡的容器，**包含TabWidget、FrameLayout两个部分**。

**TabWidget是每个Tab选项卡的标签按钮；FrameLayout是每个Tab选项卡的内容。**（TabWidget、FrameLayout可采用任意布局，例如线性布局、相对布局等）

一、定义布局文件

| 属性/方法名 |                   说明                   |
| :---------: | :--------------------------------------: |
|   TabHost   |                 自定义id                 |
|  TabWidget  |    必须为android:id="@android:id/tabs    |
| FrameLayout | 必须为android:id="@android:id/tabcontent |

二、实现

```
TabHost examTabHost=(TabHost)findViewById(R.id.examTabHost);

//在活动中加载TabHost
examTabHost.setup();

//对TabHost的选项卡操作，依次是选项卡的标签、显示方式（标签文字/图片）、对应布局
//①显示方式——文字
TabHost.TabSpec
tabSpec1=examTabHost.newTabSpec("tab1").setIndicator("单选题").setContent(R.id.single);

//②显示方式——文字+图片
TabHost.TabSpec
tabSpec1=examTabHost.newTabSpec("tab1").setIndicator(getMenuItem(R.mipmap.single,"单选题")).setContent(R.id.single);

examTabHost.addTab(tabSpec1);
```

三、选项卡显示方式（文字+图片）

setIndicator（）重载方法：动态加载布局

①创建选项卡的布局文件

②

```
 //设置需要的View对象
 public View getMenuItem(int imgID,String textID){
        LinearLayout layout=(LinearLayout) LayoutInflater.from(MainActivity.this)
                .inflate(R.layout.examtab,null);
        ImageView imageView=(ImageView)layout.findViewById(R.id.tabimg);
        imageView.setBackgroundResource(imgID);
        TextView textView=(TextView)layout.findViewById(R.id.tabtxtinfo);
        textView.setText(textID);
        return layout;
    }
```

