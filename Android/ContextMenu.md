#### ContextMenu、AlertDialog

###### ContextMenu(上下文菜单即快截菜单)

**第一种：**

①xml菜单文件

一个group里可以包含多个菜单选项

```
<menu xmlns:android="http://schemas.android.com/apk/res/android">

    <group android:id="@+id/...">
      <item
          android:id="@+id/..."
          android:title="..."/>
        <item
            .../>
        <item
            .../>
    </group>

</menu>
```

②在onCreateContextMenu()方法中使用**MenuInflater.inflate()**方法填充菜单资源，即将XML菜单资源转换成一个可编程的对象。

```
//参数含义：要加载的上下文菜单，与菜单相关的组件，菜单的附加信息
public void onCreateContextMenu(ContextMenu menu, View v, ContextMenu.ContextMenuInfo menuInfo) {

MenuInflater inflater = MainActivity.this.getMenuInflater();

//填充菜单（读取XML文件、解析、加载到Menu组件上）
inflater.inflate(R.menu.main_menu, menu);

super.onCreateContextMenu(menu, v, menuInfo);
}

```

③在onCreate()方法中填写功能代码，为View组件注册ContextMenu

长按某一组件（EditText、TextView），将其加载出来

`MainActivity.this.registerForContextMenu(editText);`

④重写onContextItemSelected()方法ContextMenu菜单建成后，需要给ContextMenu指定监听器为每个菜单项添加执行功能

```
public boolean onContextItemSelected(MenuItem item) {
        switch (item.getItemId()){
            case R.id. ..:
                ...
                break;
            case R.id.bblue:
            	break;
			...
        }
        return super.onContextItemSelected(item);
}
```

**第二种：**

①通过代码动态添加菜单项

menu.add(菜单项的组号，菜单项的ID，菜单项的排序号，菜单项标题)

（其中菜单项的排序号如果是按照菜单项的添加顺序排序，该参数的值可以都为0）

```
public void onCreateContextMenu(ContextMenu menu, View v, ContextMenu.ContextMenuInfo menuInfo) {

    menu.add(0,0,1,"背景蓝色");
    menu.add(0,1,2,"背景绿色");
    ...
    
	super.onCreateContextMenu(menu, v, menuInfo);
}
```

②在onCreate()方法中填写功能代码，为View组件注册ContextMenu

长按某一组件（EditText、TextView），将其加载出来

`MainActivity.this.registerForContextMenu(editText);`

③

```
public boolean onContextItemSelected(MenuItem item) {
        switch (item.getItemId()){
            case menu.add()中菜单项的ID:
                ...
                break;
            case ...:
            	break;
			...
        }
        return super.onContextItemSelected(item);
}
```

###### AlertDialog

AlertDialog类继承自Dialog，AlertDialog的构造方法全部都是protected的，所以不 能 直 接 通 过AlertDialog类创建一个AlertDialog对象，但是可以通过其内部类AlterDialog.Builder来创建。

例如：点击某一按钮弹出对话框

```
button.setOnClickListener(new View.OnClickListener() {
    
	public void onClick(View v) {
    	//创建对话框
        AlertDialog.Builder adialog=new AlertDialog.Builder(MainActivity.this);
        adialog.setTitle("请认真选择")	//标题
               .setMessage("你喜欢智能手机开发这门课吗？")	//内容
               .setIcon(R.mipmap.ic_launcher)	//图标
               //确定按钮
               .setPositiveButton("非常喜欢", new DialogInterface.OnClickListener() {
               		@Override
                    public void onClick(DialogInterface dialog, int which) {
						...
                    }
                    
               //列表按钮对话框
               .setItems(数据集, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                    	...         
                    }
                    
               //单选列表对话框
               .setSingleChoiceItems(数据集, 默认选中的位置（1）,new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                    	...         
                    }
                    
                //多选列表对话框
                .setMultiChoiceItems(数据集, 一个数据（标记每个选项是否默认选中）,new DialogInterface.OnMultiChoiceClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which, boolean isChecked) {

                    }          
       });
            
       adialog.create().show();
   }
});
```

