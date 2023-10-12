import numpy as np
import pandas as pd


'''
一、创建从1到10的一维数组
'''
arr1 = np.array([1,2,3,4,5,6,7,8,9,10])
# print(arr1)

'''
二、给定a = np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])，从数组中提取所有的奇数
'''
a2 = np.array([1,2,3,4,5,6,7,8,9,10])
# print(a2[a2%2 == 1])

'''
三、将a中的所有奇数替换为-1。
'''
a3 = np.array([1,2,3,4,5,6,7,8,9,10])
a3[a3%2==1] = -1
# print(a3)

'''
四、将一维数组a转换为2行的2维数组
'''
a4 = np.array([1,2,3,4,5,6,7,8,9,10])
a41 = a4.reshape(2,-1)
# print(a41)

'''
五、垂直堆叠数组a和数组b
'''
a = np.ones(10).reshape(2,-1)
b = np.zeros(10).reshape(2,-1)
c = np.vstack((a,b))
# print(c)

'''
六、水平堆叠数组a和数组b
'''
a = np.ones(10).reshape(2,-1)
b = np.zeros(10).reshape(2,-1)
c = np.hstack((a,b))
# print(c)

'''
七、获取数组a中值在5到8之间的所有项
'''
a7 = np.array([1,2,3,4,5,6,7,8,9,10])
# print(a7[(a7>=5) & (a7<=8)])

'''
八、交换数组a中的第1列和第2列
'''
a8 = np.arange(10).reshape(2,-1)
# print(a8)
a8[:,[0,1]]=a8[:,[1,0]]
# print(a8)

'''
九、从数组a创建一个Series
'''
a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])
s = pd.Series(a,index=['s1','s2','s3','s4','s5'])
# print(s)

'''
十、组a创建 DataFrame df
'''
df = pd.DataFrame(data=[b,a],index=['i0','i1'],columns=['c0','c1','c2','c3','c4'])
# print(df)

'''
十一、显示df的基础信息;包括行的数量;列名;每一列值的数量、类型
'''
# print(df.shape[0])
# print(df.columns)
# print(df.dtypes)

'''
十二、显示df的前3列
'''
# print(df.iloc[:,0:3])

'''
十三、将df的第1列从小到大排序
'''
# print(df)
df.sort_values(by='c0',inplace=True)
# print(df)

'''
十四、在df中插入一列，然后再删除这一列
'''
df.insert(3,'temp',[0,0])
# print(df)
df.drop(labels='temp',axis=1,inplace=True)
# print(df)

'''
十五、 将df中按第二列升序排列
'''
df.sort_values(by='c1',inplace=True)
# print(df)