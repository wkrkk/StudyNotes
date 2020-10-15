#### 获取EditText输入内容与Toast使用

###### 获取输入内容

`String number=edtNumber.getText().toString(); `

###### Toast显示方式

**一、直接显示**

①`Toast.makeText(getApplicationContext(),"登录成功",Toast.LENGTH_LONG).show();`

② 在strings中用键值对表示：`<string name="yes">登录成功</string>`

​     `Toast.makeText(MainActivity.this,R.string.yes,Toast.LENGTH_LONG).show();`

以上两种效果相同

**二、自定义Toast**

（1）

```
 Toast toast=new Toast(MainActivity.this);
 //定义一个ImageView
 ImageView imageView=new ImageView(MainActivity.this);
 imageView.setImageResource(R.mipmap.lock);
 //定义一个TextView
 TextView textView=new TextView(MainActivity.this)
 textView.setText("登录失败");
 //定义一个布局，这里以LinearLayout为例
 LinearLayout layout=new LinearLayout(MainActivity.this);
 //定义布局的方向
 layout.setOrientation(LinearLayout.VERTICAL);
 //将ImageView放到Layout中
 layout.addView(imageView);
 //将TextView放到Layout中
 layout.addView(textView);
 //设置View
 toast.setView(layout);
 //设置显示时间
 toast.setDuration(Toast.LENGTH_LONG);
 toast.show();
 
 选用：
 //设置控件在布局文件中的宽和高
 imageView.setLayoutParams(new LinearLayout.LayoutParams(100, 100));
 //设置字体的颜色
 textView.setTextColor(Color.parseColor("#8a8a8a"));
 //设置字体大小
 textView.setTextSize(TypedValue.COMPLEX_UNIT_SP,8);
```

（2）

①创建布局文件 .xml

②**LayoutInflater:**将xml布局资源解析并转换成View对象

```
LayoutInflater inflater=LayoutInflater.from(getApplicationContext());
View layout=inflater.inflate(R.layout.loginyes,null,false);
Toast toast=new Toast(getApplicationContext());
toast.setView(layout);
toast.show();
```

