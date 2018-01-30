
# coding: utf-8

# 参考文献: [零基础入门深度学习(1) - 感知器](https://www.zybuluo.com/hanbingtao/note/433855)

# ## 深度学习
# 
# 在人工智能领域，有一个方法叫机器学习。在机器学习这个方法里，有一类算法叫神经网络。神经网络如下图所示：
# 
# ![](http://upload-images.jianshu.io/upload_images/2256672-c6f640c11a06ac2e.png)
# 
# 上图中每个圆圈都是一个神经元，每条线表示神经元之间的连接。我们可以看到，上面的神经元被分成了多层，层与层之间的神经元有连接，而层内之间的神经元没有连接。
# * 最左边的层叫做输入层，这层负责接收输入数据；
# * 最右边的层叫输出层，我们可以从这层获取神经网络输出数据。
# * 输入层和输出层之间的层叫做隐藏层。
# 
# 隐藏层比较多（大于2）的神经网络叫做深度神经网络。而深度学习，就是使用深层架构（比如，深度神经网络）的机器学习方法。
# 
# 那么深层网络和浅层网络相比有什么优势呢？简单来说深层网络能够表达力更强。事实上，一个仅有一个隐藏层的神经网络就能拟合任何一个函数，但是它需要很多很多的神经元。而深层网络用少得多的神经元就能拟合同样的函数。也就是为了拟合一个函数，要么使用一个浅而宽的网络，要么使用一个深而窄的网络。而后者往往更节约资源。
# 
# 深度网络也有劣势，就是它不太容易训练。简单的说，你需要大量的数据，很多的技巧才能训练好一个深度网络。
# 
# ## 感知器
# 
# 为了理解神经网络，我们应该先理解神经网络的组成单元——神经元。神经元也叫做感知器。感知器算法在上个世纪50-70年代很流行，也成功解决了很多问题。并且，感知器算法也是非常简单的。
# 
# ### 感知器的定义
# 下图是一个感知器:
# ![](http://upload-images.jianshu.io/upload_images/2256672-801d65e79bfc3162.png)
# 
# 可以看到，一个感知器有如下组成部分
# * 输入权值
# * 偏置项
# * 激活函数  
# * 输出
# 
# ### 感知器还能做什么
# 
# 事实上，感知器不仅仅能实现简单的布尔运算。它可以拟合任何的线性函数，任何线性分类或线性回归问题都可以用感知器来解决。前面的布尔运算可以看作是二分类问题，即给定一个输入，输出0（属于分类0）或1（属于分类1）。如下面所示，and运算是一个线性分类问题，即可以用一条直线把分类0（false，红叉表示）和分类1（true，绿点表示）分开。
# 
# ![](http://upload-images.jianshu.io/upload_images/2256672-acff576747ef4259.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/360)
# 
# 然而，感知器却不能实现异或运算，如下图所示，异或运算不是线性的，无法用一条直线把分类0和分类1分开。
# 
# ![](http://upload-images.jianshu.io/upload_images/2256672-9b651d237936781c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/360)
# 
# ### 感知器的训练
# 现在，你可能困惑前面的权重项和偏置项的值是如何获得的。这就要用到感知器训练算法:将权重项和偏置项初始化为0，然后，利用下面的感知器规则迭代的修改w和b
# 
# 

# In[4]:


class Perception(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型是double -> double
        '''
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
        
    def __str__(self):
        '''
        打印学习到的权重，偏置项
        '''
        return 'weights\t: %s\nbias:\t: %f\n' % (self.weights, self.bias)
    
    def predict(self, input_num):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把[x1,x2,x3...]和[w1,w2,w3...]打包一起，两两相乘，求和
        return self.activator(
                reduce(lambda a, b: a+b,
                      map(lambda (x, w): x*w,
                         zip(input_num, self.weights))
                      , 0.0) + self.bias
            )
    
    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据，一组向量，对应的标签，训练轮数，学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
            
    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)
            
    def _update_weights(self, input_vec, output, label, rate):
        '''
        感知器规则更新权重
        '''
        dalta = label - output
        self.weights = map(
                lambda (x, w): w + rate*dalta*x,
                zip(input_vec, self.weights)
            )
        self.bias += rate*dalta


# 下面我们用这个感知器去实现and函数

# In[5]:


def f(x):
    '''
    定义激活函数f
    '''
    return 1 if x>0 else 0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    input_vecs = [
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1]
    ]
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perception():
    '''
    使用and真值表训练感知器
    '''
    p = Perception(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


# In[6]:


if __name__ == '__main__':
    and_perception = train_and_perception()
    print and_perception
    print '1 and 1 = %d', and_perception.predict([1,1])
    print '1 and 0 = %d', and_perception.predict([1,0])
    print '0 and 1 = %d', and_perception.predict([0,1])
    print '0 and 0 = %d', and_perception.predict([0,0])

