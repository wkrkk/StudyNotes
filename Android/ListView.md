#### ListView

**ListView**用以列表方式显示内容。

通用的两个功能：①将数据填充到布局  ②处理用户的选择点击操作

三个关键要素：①ListView中每一行的View

​                           ②填入到View中的数据（被映射的字符串、图片或基本组件）

​                           ③连接数据与ListView的Adapter

**使用示例：**

一、ArrayAdapter与ListView

- 在要显示列表的界面布局文件中添加ListView组件

  ```
  <ListView
      android:id="@+id/listview"
      android:layout_width="match_parent"
      android:layout_height="match_parent"/>
  ```

- 为ArrayAdapter装配数据

  ```
  private String [] data={"周一","周二","周三","周四","周五","周六"};
  //参数：上下文、布局资源、数据
  ArrayAdapter<String> adapter=new ArrayAdapter<String>(getApplicationContext(),android.R.layout.simple_list_item_1,data);    
  ```

| 布局资源                                          | 功能                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| android.R.layout.simple_list_item_1               | 每一项只有一个TextView                                       |
| android.R.layout.simple_list_item_multiple_choice | 每一项有一个复选框，需要用setChoiceMode（）方法设置选择模式  |
| android.R.layout.simple_list_item_single_choice   | 每一项有一个单选按钮，需要用setChoiceMode（）方法设置选择模式 |
| android.R.layout.simple_list_item_checked         | 每一项有一个选择项，需要用setChoiceMode（）方法设置选择模式  |

- 将适配器与ListView相关联

  `listview.setAdapter(adapter)`

- 设置监听事件（单击，长按）

  ```
  //参数表示：事件发生的AdapterView、点击某一项的View、该项在Adapter中的位置、该项在ListView的位置
  public void onItemClick(AdapterView<?> parent, View view, int position, long id)
  public boolean onItemLongClick(AdapterView<?> parent, View view, int position, long id)
  ```

  ```
  listview.setOnItemClickListener(new AdapterView.OnItemClickListener() {
              @Override
              public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                  //通过listView对象获取到当前listView中被选择的条目position;
                  //以下方法实现会对返回一个SparseBooleanArray集合，其中对listview的触发过点击事件的每个条目进行
                  // 标记（键值对）键==position/值==boolean，若该条目被选中则显示true，否则显示false;
                  SparseBooleanArray check=listview.getCheckedItemPositions();
                  String str="";
                  for (int i=0;i<data.length;i++){
                      if (check.get(i)){
                          str+=data[i];
                      }
                  }
                  Toast.makeText(getApplicationContext(),str,Toast.LENGTH_SHORT).show();
              }
          });
  ```

二、SimpleAdapter与ListView

- 自定义ListView每项显示的内容的布局文件

- 定义一个HashMap构成的ArrayList（列表），将数据以键值对的方式存在里面，然后构造SimpleAdapter对象，将数据装配到该适配器中。

  ```
  //参数表示：上下文、HashMap构成的列表、ListView每一行的布局文件ID、HashMap中所有键构成的字符串数组、ListView上每一行布局文件中对应组件ID构成的int型数组
  SimpleAdapter adapter=new SimpleAdapter(getApplicationContext(),arrayList,R.layout.item,from,to);
  ```

  ```
  private int [] imgIDs={R.mipmap.people,R.mipmap.people,R.mipmap.people};
  private String [] names={"Amy","Bob","Tom"};
  
  listview=(ListView)findViewById(R.id.listview);
  
  ArrayList arrayList=new ArrayList();
  //将照片、姓名使用HashMap的格式
  for(int i=0;i<imgIDs.length;i++){
         HashMap map=new HashMap();
         map.put("img",imgIDs[i]);
         map.put("name",names[i]);
         arrayList.add(map);
  }
    
  String []from={"img","name"};
  int [] to={R.id.listview_img,R.id.listview_name};
  
  SimpleAdapter adapter=new SimpleAdapter(getApplicationContext(),arrayList,R.layout.item,from,to);
  listview.setAdapter(adapter);
  ```

  