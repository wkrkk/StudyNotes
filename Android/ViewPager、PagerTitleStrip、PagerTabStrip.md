#### ViewPager、PagerTitleStrip、PagerTabStrip

**ViewPager**是Android3.0后引入的一个UI组件，可以通过手势滑动来完成View的切换，比如作为App的引导页、实现图片轮播等功能。ViewPager是android-support-v4.jar中定义的类，它直接继承了ViewGroup类，它也是一个容器类，可以在其中添加其他的View类。

（TabLayout可以与ViewPager联用，实现点击或滑屏效果）

为了实现在ViewPager中显示子一级View对象，就需要**PageAdapter**协调来完成。创建一个类继承PageAdapter，并重写里面的方法。

```
    private ArrayList<View> viewLists;
    public MyPAdapter() {
    }
    public MyPAdapter(ArrayList<View> viewLists) {
        this.viewLists=viewLists;
    }

    //将给定position位置的view添加到container中，并创建后显示出来；
    // 返回一个代表新增页面的Object(key)，通常都是直接返回View，也可以自定义key，但是key和View要一一对应。
    public Object instantiateItem(@NonNull ViewGroup container, int position) {
        //把从list中加载的View添加到当前的ViewPager中
        container.addView(viewLists.get(position));
        //创建后显示出来
        return viewLists.get(position);
    }

    //移除一个给定position位置的页面
    public void destroyItem(@NonNull ViewGroup container, int position, @NonNull Object object) {
       //移除从list当前加载的View
       container.removeView(viewLists.get(position));
    }

    //获得ViewPager中有View对象的个数
    public int getCount() {
        return viewLists.size();
    }

    //判断instantiateItem()方法所返回来的key与一个页面视图是否代表的同一个View
    // (即它俩是否是对应的，对应的表示同一个View)
    // 通常直接写“return view == object”代码
    public boolean isViewFromObject(@NonNull View view, @NonNull Object o) {
        return view==o;
    }
```

**PagerTitleStrip**是ViewPage的一个关于当前页面、上一个页面和下一个页面的非交互性指示器（相当于每个页面的标题）。使用时一般将它作为ViewPage的子级组件。（实现方法与子View类似，在继承PageAdapter类中添加方法public CharSequence getPageTitle(int position) ）

**PagerTabStrip**是ViewPage的一个关于当前页面、上一个页面和下一个页面的交互性指示器。其余上同。