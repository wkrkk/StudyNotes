#### TableLayout、CountDownTimer的使用

###### TableLayout

表格布局，一行放置多个组件时，可以使用TableRow

```
//必选属性
android:stretchColumns="*" //允许被拉伸的列序号
android:shrinkColumns="*"  //允许被收缩的列序号
```

###### CountDownTimer

创建一个继承CountDownTimer的内部类，并重写相关方法

`new myCount(60*1000,1000).start();`

```
 private class myCount extends CountDownTimer{

        构造方法：
        millisInFuture表示从开始调用start()到计时结束调用onFinish()的总时间
        countDownInterval表示调用onTick()的间隔时间
        * */
        public myCount(long millisInFuture, long countDownInterval) {
            super(millisInFuture, countDownInterval);
        }

        //每个间隔时间一到，就会调用该方法，millisUntilFinished表示剩余时间
        @Override
        public void onTick(long millisUntilFinished) {
            long hour=millisUntilFinished/1000/3600;//时
            long minute=millisUntilFinished/1000%3600/60;//分
            long second=millisUntilFinished/1000%3600%60;//秒
            tvTime.setText("倒计时："+hour+":"+minute+":"+second);
            ...

        }

        @Override
        public void onFinish() {
            tvTime.setText("计时结束！");
            ...

        }
    }
```

