import tensorflow as tf
import numpy as np

# 输入数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 生成300个-1到1之间的均匀数字，形成数组。new axis放在后面，增加数组维数，变成300x1的矩阵
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5+noise

# 输入层
with tf.name_scope('input_layer'):  # 输入层。将这两个变量放到input_layer作用域下，tensor board会把他们放在一个图形里面
    # 注意这里的with操作完成之后“input_layer”名字还是存在内存中。如果下面再次使用with tf.name_scope('input_layer')，会生成"input_layer_1"
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # xs起名x_input，会在图形上显示--placeholder是一个占位操作，操作名为x_input
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')  # ys起名y_input，会在图形上显示

# 隐层
with tf.name_scope('hidden_layer'):  # 隐层。将隐层权重、偏置、净输入放在一起
    with tf.name_scope('weight'):  # 权重
        W1 = tf.Variable(tf.random_normal([1, 10]))  # 定义一个1x10的矩阵变量，需要给初始值，这里值服从正态分布。
# 注意sess.run(W1)之后才有值。变量可以通过assign 函数修改
        tf.summary.histogram('hidden_layer/weight', W1)  # 生成直方图名称：“hidden_layer/weight”，值：W1
    with tf.name_scope('bias'):  # 偏置
        b1 = tf.Variable(tf.zeros([1, 10])+0.1)
        tf.summary.histogram('hidden_layer/bias', b1)
    with tf.name_scope('Wx_plus_b'):  # 净输入
        Wx_plus_b1 = tf.matmul(xs, W1) + b1
        tf.summary.histogram('hidden_layer/Wx_plus_b', Wx_plus_b1)
output1 = tf.nn.relu(Wx_plus_b1)

# 输出层
with tf.name_scope('output_layer'):  # 输出层。将输出层权重、偏置、净输入放在一起
    with tf.name_scope('weight'):  # 权重
        W2 = tf.Variable(tf.random_normal([10, 1]))
        tf.summary.histogram('output_layer/weight', W2)
    with tf.name_scope('bias'):   # 偏置
        b2 = tf.Variable(tf.zeros([1, 1])+0.1)
        tf.summary.histogram('output_layer/bias', b2)
    with tf.name_scope('Wx_plus_b'):  # 净输入
        Wx_plus_b2 = tf.matmul(output1, W2) + b2
        tf.summary.histogram('output_layer/Wx_plus_b', Wx_plus_b2)
output2 = Wx_plus_b2

# 损失
with tf.name_scope('loss'):  # 损失
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-output2), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):  # 训练过程
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化
init = tf.global_variables_initializer()  # 定义初始化的操作
sess = tf.Session()  # 建立Session
sess.run(init)  # 执行初始化操作
merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
writer = tf.summary.FileWriter('logs', sess.graph)  # 将训练日志写入到logs文件夹下

# 训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if((i % 50) == 0):  # 每50次写一次日志
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})  # 计算需要写入的日志数据
        #  将日志数据写入文件
        writer.add_summary(result, i)
