#### OptionsMenu、PopupMenu、DatePickerDialog

###### OptionsMenu

①设置OptionsMenu菜单，重写onCreateOptionsMenu()方法；

//将其显示在ActionBar中

`menu.add(..).setShowAsAction(MenuItem.SHOW_AS_ACTION_ALWAYS);`

②实现OptionsMenu每一项的点击效果，重写onOptionsItemSelected()方法。

###### PopupMenu

内部类实现

```
PopupMenu popupMenu=new PopupMenu(MainActivity.this,edtCity);                popupMenu.getMenuInflater().inflate(R.menu.menu_pop,popupMenu.getMenu());

popupMenu.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
	@Override
    public boolean onMenuItemClick(MenuItem item) {
        ...            
    	return false;
    }
});

popupMenu.show();
```

###### DatePickerDialog、TimePickerDialog

```
//获取当前系统时间
Calendar calendar=Calendar.getInstance();
int mYear=calendar.get(Calendar.YEAR);
int mMonth=calendar.get(Calendar.MONTH);
int mDay=calendar.get(Calendar.DAY_OF_MONTH);
```

```
DatePickerDialog datePickerDialog=new DatePickerDialog(MainActivity.this, DatePickerDialog.THEME_DEVICE_DEFAULT_LIGHT, new DatePickerDialog.OnDateSetListener() {
	@Override
	public void onDateSet(DatePicker view, int year, int month, int dayOfMonth) {

	}
},mYear,mMonth,mDay);
datePickerDialog.show();
```

