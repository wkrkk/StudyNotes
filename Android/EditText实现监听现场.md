#### EditText实现监听现场

###### **setOnEditorActionListener方法的使用**

<u>编辑完之后点击软键盘上的各种键才会触发</u>

①在布局文件中的EditView控件中设置以下属性：

```
android:imeOptions="..."
android:inputType="...“
```

- imeOptions=”actionUnspecified” –> EditorInfo.IME_ACTION_UNSPECIFIED
- imeOptions=”actionNone” –> EditorInfo.IME_ACTION_NONE
- imeOptions=”actionGo” –> EditorInfo.IME_ACTION_GO
- imeOptions=”actionSearch” –> EditorInfo.IME_ACTION_SEARCH
- imeOptions=”actionSend” –> EditorInfo.IME_ACTION_SEND
- imeOptions=”actionNext” –> EditorInfo.IME_ACTION_NEXT
- imeOptions=”actionDone” –> EditorInfo.IME_ACTION_DONE

②监听事件执行

```
edtContent.setOnEditorActionListener(new EditText.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                if(actionId == IME_ACTION_DONE){
                   ...
                   ...
                }
                return false; 
            }

        });
```

###### 