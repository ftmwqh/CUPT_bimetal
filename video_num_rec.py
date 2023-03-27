import numpy
import matplotlib.pyplot as plt
import glob
import scipy.special
import imageio
from PIL import Image
import cv2
import os

# 下面是2层神经网络模型
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 下面所定义的inodes、hnodes、onodes和lr就是只在这个neuralNetwork类的里面使用，就相当于内部参数，使用的时候直接使用self来进行
        # 调用
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # rand()生成[0,1)的数据，另外要注意W矩阵的行和列数，这里的wih和who直接就是我们之前所说的w的转置，
        # 即这里的wih和who是不需要再进行任何的转置操作了，他们可以直接和输入x相乘
        # self.wih = numpy.random.rand(self.hnodes,self.inodes) - 0.5
        # self.who =  numpy.random.rand(self.onodes,self.hnodes) - 0.5

        # 下面是更为复杂的初始参数的设置
        '''
        我们将正态分布的中心设定为0.0。 与下一层中节点相关的标准方差的表达式， 按照Python的形式， 就是pow(self.hnodes, -0.5)， 
        简单说来， 这个表达式就是表示节点数目的-0.5次方。 最后一个参数， 就是我们希望的numpy数组的形状大小。
        '''
        # 下面定义的wih和who也是只在当前这个类中使用，相当于就是一个内部参数，使用的时候就直接使用self来进行调用
        # 初始化各层的权重系数
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))  # 使用正态概率分布采样权重， 其中平均值为0， 标准方差为节点传入链接数目的开方，
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)  # 定义激活函数
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T  # 把输入的数据变成列向量
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)  # 计算隐藏层的输入，也就是计算输入层和隐藏层之间的权重系数与输入数据的乘积

        hidden_outputs = self.activation_function(hidden_inputs)  # 使用激活函数计算隐藏层的输出

        final_inputs = numpy.dot(self.who, hidden_outputs)  # 计算输出层的输入，也就是计算隐藏层和输出层之间的权重系数与输入数据的乘积

        final_outputs = self.activation_function(final_inputs)  # 使用激活函数计算输出层的输出

        output_errors = targets - final_outputs  # 计算误差，这个误差是最开始的误差，也就是目标值和输出层输出的数据的差

        hidden_errors = numpy.dot(self.who.T, output_errors)  # 这是输出层到隐藏层之间的误差反向传播

        # 下面是利用误差的反向传播来更新各层之间的权重参数
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))  # 这是更新隐藏层和输出层之间的权重参数

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))  # 这是更新输入层和隐藏层之间的权重参数
        return self.wih,self.wih
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)  # 计算隐藏层的输入，也就是计算输入层和隐藏层之间的权重系数与输入数据的乘积

        hidden_outputs = self.activation_function(hidden_inputs)  # 使用激活函数计算隐藏层的输出

        final_inputs = numpy.dot(self.who, hidden_outputs)  # 计算输出层的输入，也就是计算隐藏层和输出层之间的权重系数与输入数据的乘积

        final_outputs = self.activation_function(final_inputs)  # 使用激活函数计算输出层的输出

        return final_outputs
        pass

class trained_neural():
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate,wih,who):
        # 下面所定义的inodes、hnodes、onodes和lr就是只在这个neuralNetwork类的里面使用，就相当于内部参数，使用的时候直接使用self来进行
        # 调用
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)  # 定义激活函数
        self.wih=wih
        self.who=who
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)  # 计算隐藏层的输入，也就是计算输入层和隐藏层之间的权重系数与输入数据的乘积

        hidden_outputs = self.activation_function(hidden_inputs)  # 使用激活函数计算隐藏层的输出

        final_inputs = numpy.dot(self.who, hidden_outputs)  # 计算输出层的输入，也就是计算隐藏层和输出层之间的权重系数与输入数据的乘积

        final_outputs = self.activation_function(final_inputs)  # 使用激活函数计算输出层的输出

        return final_outputs
        pass
# 下面input_nodes、hidden_nodes、output_nodes和learning_rate就相当于我们在主函数中定义的变量一样，只在类之外的主函数中使用
# 设置各层的神经元个数以及学习率，注意这里是设置各层的神经元的个数，而不是各层的层数，我们这个程序只是一个最简单的2层神经网络模型
input_nodes = 784  # 输入层有784个神经元，也就是我们共有784个数据需要输入给神经网络
hidden_nodes = 100  # 隐藏层有100个神经元，注意这里是隐藏层节点的个数，而不是隐藏层的层数，我们这里隐藏层就只有一层，也就是2层神经网络模型。增大隐藏层节点个数，代码运行时间明显加长
output_nodes = 10  # 输出层有10个神经元，每一个神经元的输出最大值都对应着当前神经网络所预测到的数字
learning_rate = 0.3  # 增大学习率，代码运行的总时间好像没有减少啊，但是正确率的确下降了

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


'''f=open('wih.txt','r')
lines=f.readlines()
a=len(lines)
print('height=',a)
b=len(lines[1].split(' '))
print('width=',b)
w1=numpy.zeros((a,1),dtype=float)

i=0
for line in lines:
    list=line.strip('\n').split(' ')
    w1[i]=list[0:b]
    i+=1
w1=w1.reshape(100,784)
f=open('who.txt','r')
lines=f.readlines()
a=len(lines)
print('height=',a)
b=len(lines[1].split(' '))
print('width=',b)
w2=numpy.zeros((a,1),dtype=float)

i=0
for line in lines:
    list=line.strip('\n').split(' ')
    w2[i]=list[2:n]
    i+=1
w2=w2.reshape(10,100)
tn=trained_neural(input_nodes, hidden_nodes, output_nodes, learning_rate,w1,w2)'''#尝试使用已训练的神经
# 注意下面打开的是训练集，这里面只有100个训练样本
echo = 3  # 当echo=1时，准确率达不到1，但是当echo等于3时，准确率基本上一直是1
for e in range(echo):  # 对同一个数据集进行多个世代的训练，也可以增加正确率
    training_data_file = open("mnist_train.csv",
                              'r')  # 打开训练数据所在文件，这个训练集中共有100个手写数字，数字是0-9，即每一个数字会有好几种不同的写法在里面
    training_data_list = training_data_file.readlines()  # 按行读取训练数据
    training_data_file.close()  # 关闭文件

    for record in training_data_list:
        # print(record)#这里面的record就是training_data_list训练集中的每一行数据，使用for循环来遍历整个训练集
        all_values = record.split(',')  # 使用逗号 , 来进行分割，每隔一个 ，就分割出一个字符串元素，每循环一次record都会更新一次，更新成最新的训练样本，总共有100个训练样本
        # asfarry表示把文本字符串转换实数，注意此时的输入是一个1*784的数组，这里并不需要把数据变成28x28的矩阵，只有画图的时候才需要把数据变成矩阵
        inputs = (numpy.asfarray(
            all_values[1:]) / 255.0 * 0.99) + 0.01  # 把数据大小范围限定在0.01-1.01之间，我们要避免让0作为神经网络的输入值，所以这里要进行+0.01的操作
        # 为了防止0和1会导致大的权重和饱和网络，我们把最后10的输出值（取0和1两个值）变成了取0.01和0.99
        targets = numpy.zeros(output_nodes) + 0.01
        # print(targets)#此时的targets的大小是100*10，100的意思是指训练集中总共有100个样本，10的意思是指每一个训练样本的的输出都是1行10列的矩阵，哪一列的数最大，那么神经网络的预测数字值就是对应的列数
        targets[int(all_values[0])] = 0.99  # 这句话的意思是把训练集中的每一个样本的第一个数据（也就是标签值）都改成了0.99，防止出现1
        wih,who=n.train(inputs, targets)
    pass
pass
'''wih=numpy.array(wih)
who=numpy.array(who)
numpy.savetxt('wih.txt',wih)
numpy.savetxt('who.txt',who)'''
test_data_file = open("mnist_train_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:

    all_values = record.split(',')

    correct_label = int(all_values[0])
    print(correct_label, "correct label")

    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    outputs = n.query(inputs)


    label = numpy.argmax(outputs)
    print(label, "network's answer")

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass
pass
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

#注意下面打开的是测试集，这里面有60000条数据
'''test_data_file = open("mnist_train.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
all_values = test_data_list[3].split(',')#这一行决定着我们要把数据集中的哪一行作为测试样本
#下面这行的作用是把测试样本的数据进行改造，即把他们从索引值为1开始向后取，然后把字符串换成实数，最后再把这些数据变成28x28的矩阵
#注意只有在画图的时候才需要把数据变成28x28的矩阵，如果只是作为神经网络的输入，那么就是一长串的数字（28x28=784个数字）
image_array = numpy.asfarray( all_values[1:]).reshape((28,28))#[1:]表示取出来除了第一个元素之外的所有值，也就是从索引值为1的那个元素开始取，因为第一个是标签值，暂时不需要
#print(image_array)

#测试训练好的神经网络
test_inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
n.query(test_inputs)

plt.imshow( image_array, cmap='Greys',interpolation='None')'''
#plt.show()
'''

'''
#多个世代训练同一个数据集
echo = 3
for e in range(echo):

    pass

def Square_Ima(read_file):
    image=Image.open(read_file)
    image=image.convert('RGB')
    w,h=image.size
    new_image=Image.new('RGB',size=(max(w,h),max(w,h)),color='black')
    length=int(abs(w-h)//2)
    box=(length,0) if w<h else (0,length)
    new_image.paste(image,box)
    new_image=new_image.resize((28,28))
    new_image.save(read_file)
    #new_image.show()
#测试一张图片识别效果
Square_Ima('2.jpg')
'''
im=Image.open('8.jpg')
im.thumbnail((28,28))
im.save('8.jpg')'''
img_array = imageio.imread('2.jpg', as_gray=True)
img_array=0+(img_array>150)*254
print(img_array)
pic=Image.fromarray(img_array)
#pic.show()
    # reshape from 28x28 to list of 784 values, invert values
img_data = img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
#print(img_data)
#print(numpy.min(img_data))
#print(numpy.max(img_data))

# append label and image data  to test data set
record = numpy.append(label, img_data)
inputs=record[1:]
outputs=n.query(inputs)
print(outputs)
label = numpy.argmax(outputs)
print("network says ", label)

def read_file_pics(path_name):
    path=path_name
    path_list=os.listdir(path)

    path_list.sort(key=lambda x:int(x[:-4]))
    print(path_list)
    i=0
    I=numpy.zeros(len(path_list))
    for filename in path_list:
        Square_Ima(os.path.join(path,filename))
        img_array = imageio.imread(os.path.join(path,filename), as_gray=True)
        img_array = 0 + (img_array > 130) * 254
        #print(img_array)
        pic = Image.fromarray(img_array)
        #pic.show()
        img_data = img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01
        outputs=n.query(img_data)
        result=numpy.argmax(outputs)
        I[i]=float(result)
        #print(result)
        i+=1
    return I

U55C30W_1_1 = read_file_pics(r'E:\py_projects\bimetal_vib\55C30W_1_1')
print(U55C30W_1_1)
U55C30W_1_2=read_file_pics(r'E:\py_projects\bimetal_vib\55C30W_1_2')
print(U55C30W_1_2)