# mxnet faster rcnn
-------------------
- 将rcnn目录拷贝到mxnet/example/下
- 模型训练：
    + bash srcipt/vggm_hcar.sh 0
- 模型测试：
    + python hcar.py --network vggm --prefix model/e2e --epoch 20 --in_dir data/hcar/JPEGImages/ --out_dir output/ --label_dir data/hcar/Annotations/ --gpu 0 --test test.txt
    + python hcar.py --network vggm --prefix model/nvggm-2 --epoch 20 --in_dir data/hcar/yolo/data2/shitai7cutpeople_test_img/ --out_dir output/ --with_label 0  --gpu 0 --test test2.txt
    + python hcar.py --network vggm --prefix model/nvggm-noise --epoch 6 --in_dir data/hcar/rcnn-format/c5 --out_dir output/ --with_label 1  --label_dir data/hcar/rcnn-format/c5 --gpu 0 --test data/hcar/rcnn-format/c5/train.txt
以下文件均需要在mxnet/example/rcnn/下运行
- KCF+RCNN视频内多目标追踪，KCF并行：
	python combination.py ./视频文件名
- 延时播放RCNN批处理：
	python batch.py ./视频文件名
- 多路视频：
	python multi.py ./视频文件名
- 解决少检测最后一秒和并行造成的kcf运算结果异常的bug：
	python mul.py ./视频文件名
- 更改内置预存视频维护方式，kcf和faster-rcnn比较函数可选：
	python mul_q.py ./视频文件名
- 多线程处理视频单rcnn网络，原本打算将创建网络的函数放置在视频处理函数外，将创建的网络作为参数传递给函数，可能由于共享网络有问题失败中无法创建视频框，当在进程函数外创建predictor传入函数时执行到mx.nd.array函数无法继续执行原因不明，采用了另外的方式实现单网络多视频，将视频关键帧预存，再传递给同一个网络，网络处理结果并行输出，mul_s.py按第二个方法实现：
	python mul_s.py ./视频文件名
- 调用java：
	+ export JAVA_HOME=/home/tracking/usr/src/jdk1.8.0_141
	+ export PATH=$JAVA_HOME/bin:$PATH
	+ export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
- 检查java：
	+ java -version
- 使用java调用python内容：
	+ 使用包org.python都调用python脚本文件或函数，或使用Runtime.getRuntime()执行脚本文件，invoke.java为java调用mul_q.py的demo
rcnn第一次调用慢，每次调用第一个画面处理慢，cv2.waitKey()函数调用非常慢大约为0.1s，生成kcf进程和储存待检测帧较慢，由于提前播放并缓存没有视频流使用缓存下来的帧时需关闭cap否则极其缓慢

