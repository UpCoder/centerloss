- 本文主要讲解自己对CenterLoss的一些理解，想要看原文的请戳这里
	[A discriminative feature learning approach for deep face recognition](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31)
- background
	- CenterLoss提出的主要目的是对[FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)的改进，FaceNet使用的是triple loss，该计算方法需要我们提前计算出三元组，计算量大不说，而且收敛很慢。
	- 所以CenterLoss就被提出来了
- 自己的理解
	- 其实CenterLoss的想法也很简单，感觉和聚类有异曲同工之妙。
	- 常用的Softmax loss一般只关注于让我们不同的类可以被正确的被分开就可以了，比如下图所示的MNIST数据集使用softmaxloss 训练，提取出来features可视化之后的结果。
	 ![image](http://ocnsbpp0d.bkt.clouddn.com/pReLU.png)
 - 我们可以看到，虽然的确不同的类别被分到了不同的地方（cluster），但是每个cluster之间距离较近，而且类内的差距较大，也就是说我们得到的features并不能通过一个较为简单的分类器将其区分开来，需要一个相对较为复杂的分类器才可以得到较好的结果。也就是说，我们得到的features并不是一个很好的特征。
 - 那么怎么改进呢？就像我们上面所说，只要让每个cluster之间的距离相对来说远一些，类内的差距小一下，那么区分起来应该就更容易了。
 - 那么具体来说怎么实现让各个cluster之间的距离远一些，类内的差距小一些呢？我们可以考虑下面的loss函数
$$
\mathcal{L}_C = \frac{1}{2}\sum_{i=1}^m||x_i - c_{y_i}||_2^2
$$
	- 其中$x_i$是我们MNIST数据集中样本对应的features，$c_{y_i}$对应的是第$y_i$个中心，其中$i \in \{1,2,...,10\}$
	- 通过上述函数，我们自然就约束了同一个类别中的样本到质心的距离，这样就可以让同一个cluster里面的数据更加聚集。
	- 文章中也给出了更新质心的公式
	$$
\bigtriangleup c_j = \frac{\sum_{i=1}^m\delta(y_i== j)*(c_j - x_i)}{1 + \sum_{i=1}^m \delta(y_i == j)}\\
c_j = c-\bigtriangleup c_j
$$
		- 其中$\delta(y_i == j)$是判断第i个sample的label是不是等于j，如果是则返回1，否则返回0
		- $c_j$代表的就是第j个类别的质心
		- $x_i$代表的是第i个样本
	- 最终我们优化的target是$\mathcal{L} = \mathcal{L_s} + \lambda * \mathcal{L_c}$
- 实现
	- 第一点：怎么计算CenterLoss
		- 一种是我自己写的，感觉很蠢。。。
		

			```python
			def calculate_centerloss(x_tensor, label_tensor, centers_tensor, category_num):
					    loss_tensor = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)
					    for i in range(category_num):
					        selected_x = tf.gather(x_tensor, 
					                               tf.cast(
					                                   tf.squeeze(
					                                       tf.where(
					                                           tf.equal(
					                                               label_tensor, 
					                                               [i]
					                                           ), 
					                                           None, 
					                                           None
					                                       )
					                                   ),tf.int32))
					        selected_y = tf.convert_to_tensor(centers_tensor[i], dtype=tf.float32)
					        selected_x2 = tf.multiply(selected_x, selected_x)
					        selected_y2 = tf.multiply(selected_y, selected_y)
					        selected_xy = tf.multiply(selected_x, selected_y)
					        distance_category = tf.abs(tf.subtract(tf.add(selected_x2, selected_y2), 2*selected_xy))
					        distance_category = tf.abs(tf.reduce_sum(0.5 * distance_category))
					        loss_tensor = tf.abs(tf.add(loss_tensor, distance_category))
					    return loss_tensor
			```
	
	
		- 还有一种就两行代码，其实他的center loss其实就是l2 loss,所以可以通过下面的代码计算
	
			```python
			def calculate_centerloss(x_tensor, label_tensor, centers_tensor):
			    centers_tensor_batch = tf.gather(centers_tensor, label_tensor)
			    loss = tf.nn.l2_loss(x_tensor - centers_tensor_batch)
			    return loss
			```
		- 大家有兴趣可以跑一跑，两个函数的输出是一样的，不过明显下面的简洁很多
	- 第二点：怎么更新Center?
		- 这里面牵扯两点，1、怎么计算Center更新的值；2、怎么在每次反向传播的时候执行操作1。
		- 先看第一点，怎么计算center的更新值


			```python
			def update_centers(centers, data, labels, category_num):
			    centers = np.array(centers)
			    data = np.array(data)
			    labels = np.array(np.argmax(labels, 1))
			    centers_batch = centers[labels]
			    diff = centers_batch - data
			    for i in range(category_num):
			        cur_diff = diff[labels == i]
			        cur_diff = cur_diff / (1.0 + 1.0 * len(cur_diff))
			        cur_diff = cur_diff * _alpha
			        for j in range(len(cur_diff)):
			            centers[i, :] -= cur_diff[j, :]
			    return centers
			```
		- 接下来我们关注第二点，怎么在每次反向传播的时候执行上述代码
			- tensorflow里面提供了py_func这个函数，通过该函数将普通的python function转化为可以在Graph 上执行的op，该函数的定义是：

				```python
				tf.py_func(func, inp, Tout, stateful=True, name=None)
				```
				func就是我们上述的update_centers, inp是tensor类型的变量，Tout是tf.float32(注意，update_centers里面接收到的参数是numpy数组)
		- 然后在每次训练的时候，也执行该函数返回，如下面所示

			```python
			owner_step = tf.py_func(update_centers, [centers_tensor, featuresmap, label_tensor, category_num], tf.float32)
			# 在每次更新的时候也执行我们自己的step
			_, centers_value, = sess.run(
			            [train_op, owner_step], feed_dict={
			                images_tensor: train_images_batch,
			                label_tensor: train_labels_batch,
			                is_training_tensor: True,
			                centers_tensor: centers_value
			            })
			```

- 结果展示
	- 下面分别是不使用CenterLoss和$\lambda$分别等于0.01， 0.1， 1.0的结果
	![image](http://ocnsbpp0d.bkt.clouddn.com/pReLU.png)
		![image](http://ocnsbpp0d.bkt.clouddn.com/Training_LS_LC_0.01_16000.png)
			![image](http://ocnsbpp0d.bkt.clouddn.com/Training_LS_LC_0.1_16000.png)
				![image](http://ocnsbpp0d.bkt.clouddn.com/Training_LS_LC_1.0_25000.png)
- 具体实现代码：[CenterLoss](https://github.com/UpCoder/centerloss)
- [博客地址](http://blog.csdn.net/liangdong2014/article/details/79076094)
