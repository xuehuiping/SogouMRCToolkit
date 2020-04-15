date: 2020/4/14 5:34 下午 

1. 我去掉了gpu相关的配置，在setup.py里面


2. 需要准备spacy的模型文件
```
下载en_core_web_sm-2.2.5.tar，并解压
/Users/huihui/Downloads/en_core_web_sm-2.2.5.tar.gz
本机目录：/Users/huihui/git/SogouMRCToolkit/env/lib/python3.7/site-packages/spacy-2.2.4-py3.7-macosx-10.14-x86_64.egg/spacy/data/en_core_web_sm
将en_core_web_sm目录拷贝到目录env/lib/python3.6/site-packages/spacy/data/
并重命名为en
```

3. 在ubuntu机器，可以训练bert+squad
```
2020-04-14 17:58:06 开始训练模型
(env) xuehp@haomeiya002:~/git/SMRCToolkit$ nohup python examples/run_bert/run_bert_squad.py  &

pid 15588

```

error 

`
2020-04-14 18:23:37.955468: W tensorflow/core/framework/allocator.cc:107] Allocation of 750108672 exceeds 10% of system memory.
`

训练
重启机器之后，再次训练
```
2020-04-15 11:10:31
(env) xuehp@haomeiya002:~/git/SMRCToolkit$ nohup python examples/run_bert/run_bert_squad.py  &
pid 4197

epochs=2
2020-04-15 13:54:51 训练完毕
2020-04-15 13:32:03,304 - root - INFO - - Train metrics: loss: 3.794
2020-04-15 13:40:43,113 - root - INFO - - Eval metrics: loss: 2.950
2020-04-15 13:40:51,627 - root - INFO - - Eval metrics: exact_match: 24.437 ; f1: 36.652
2020-04-15 13:40:51,628 - root - INFO - - epoch 1 eposide 1: Found new best score: 36.652091
2020-04-15 13:40:52,659 - root - INFO - - Found new best model, saving in models/bert/squad/best_weights/after-eposide-1

大约一个小时能运行1个epoch
```

训练
修改epoch=20，重新训练
pid=5801

4. 测试bert+squad