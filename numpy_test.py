import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
mylist=[[2,3,-5],[21,-2,1]]
a=N.array(mylist,dtype='d')
b=N.zeros((3,2,4),dtype='d')
c=N.arange(10)
d=[1,2,3,4,5]
e=N.zeros((4,5,2),dtype='d')
f=N.array([[2,3.2,5.5,-6.4,-2.2,2.4],
			[1,22,4,0.1,5.3,-9],
			[3,1,2.1,21,1.1,-2]])
g=N.arange(6)
h=N.arange(8)
i=N.zeros((3,4),dtype='d')
print(N.reshape(g,(2,3)))
print(N.transpose(i))
N.ravel(g)
print(g)
print(N.concatenate((g,h)))
print(N.repeat(g,3))
j=g.astype('f')
print(j)
'''
'''
lon=N.array([0,45,90,135,180,225,270,315,360])
lat=N.array([-90,-45,0,45,90])
a=N.meshgrid(lon,lat)
print(a[1])
'''
##两数组对应元素相乘:方法一
'''
a=N.array([[2,3.2,5.5,-6.4],
[3,1,2.1,21]])
b=N.array([[4,1.2,-4,9.1],
[6,21,1.5,-27]])
shape_a=N.shape(a)
print(shape_a)
product_ab=N.zeros(shape_a,dtype='f')
for i in range(shape_a[0]):
	for j in range(shape_a[1]):
		product_ab[i,j]=a[i,j]*b[i,j]
print(product_ab)
'''
'''
#方法二
a=N.array([[2,3.2,5.5,-6.4],
[3,1,2.1,21]])
b=N.array([[4,1.2,-4,9.1],
[6,21,1.5,-27]])
c=N.zeros(N.shape(a),dtype='f')
print(N.size(a))
#for i in range(N.size(a)):
product_ab=a*b
print(product_ab)
'''
'''
a=N.arange(10)
b=a*2
c=a+b
d=c*2.0

print(a)
print(b)
print(c)
print(d)
'''
'''
#计算位温
def theta(p,T,p0=1000.0,kappa=0.286):
	return T*(p0/p)**(kappa)
p=N.array([1020.0,1013.5,1017.2])
T=N.array([273.1,280.4,276.6])
print(theta(p,T))
'''
'''
a=N.arange(8)
print(a>3)
print(N.greater(a,3))
print(N.logical_and(a>1,a<=3))
condition=N.logical_and(a>3,a<6)
answer=N.where(condition,a*2,0)
#print(answer)
#对数组进行切片
answer_indices=N.where(condition)
answer_1=(a*2)[answer_indices]
print(answer_indices)
print(answer_1)

a=N.reshape(N.arange(8),(2,2,2))
condition=N.logical_and(a>3,a<6)
answer_indices=N.where(condition)
answer=(a*2)[answer_indices]
print(condition)
print(a)
print(answer_indices)
print(answer)

a=N.arange(8)
condition=N.logical_and(a>3,a<6)
print(condition*1)
print(N.logical_not(condition))
'''
'''
#计算程序运行需要的时间
import time
begin_time=time.time()
for i in range(1000000):
	a=2*3
print(time.time(),begin_time)
print(time.time()-begin_time)
'''
'''
#大于临界风速的不作处理，小于临界风速的做近似为最小风速处理
def good_magnitudes(u,v,minmag=0.1):
	mag=(u**2+v**2)**0.5
	output=N.where(mag>minmag,mag,minmag)
	return output
'''
#初窥我python中的ncl
'''
#read nc
import Ngl,Nio
import numpy as N
fname="/home/tangyuheng/data_ncl/SH_80_mon.nc"
f=Nio.open_file(fname)
print(f.variables)
SH=f.variables['sh']
#SH_arr=N.array(SH)
print(N.shape(SH))
print(SH[:,0,2])
ID=f.variables['id']
print(ID)
'''
'''
import my_lib
import numpy as np
var=12
print(type(var))
var=float(var)
print(type(var))
var=str(var)
print(type(var))
array=[0,3,6,9]
array=np.array(array)
print(type(array),array)
array[:]=0
print(array)
a=np.array([1,2,3,4])
b=np.array([0,1,1,0])
c=a+b
print(c,type(c))
n=np.zeros((4,5),int)
print(n,np.shape(n))
l=np.full(100,1.0e20)
print(l,type(l))
a=np.array([1,2,3,4])
a_rev=a[::-1]
print(a_rev)
b=np.array([1,-999,3,4,5,-999,7,-999])
ind_not0=np.nonzero(b!=-999)
print(ind_not0)
print(b[ind_not0])
i=np.arange(0,11,1)
print(i)
#lat=np.arange(-180.0,181.0,30.0)
lat=np.linspace(-180.0,180.0,13)
print(lat)
ra=np.array([[[1,2,3,4],[5,6,7,8],[5,4,2,0]],\
[[1,2,3,4],[5,6,7,8],[5,4,2,0]]])
print(np.shape(ra),ra[1,:,0])
rald=np.ravel(ra)
print(rald,rald.shape)
print(ra.shape)
ra3d=np.reshape(rald,ra.shape)
print(ra3d)
ranm=np.where(ra==0,-999,ra)
print(ranm)

t=99
if t==0:
	print('t=0')
elif t==1:
	print('t=1')
else:
	print('t={}'.format(t))
'''
import Ngl,Nio
'''
#read nc绘图
import Ngl,Nio
import numpy as np
fname="/home/tangyuheng/data_ncl/hgt.1986.nc"
f=Nio.open_file(fname)
print(f.variables)
print(f.variables['hgt'])#365*17*73*144
print(f.variables['level'][5])#500hPa
var=f.variables['hgt'][189,5,:,:]
lat=f.variables['lat'][:]
lon=f.variables['lon'][:]
var=var+31265
var=np.array(var,dtype='int')
print(type(var[50,50]))
wks=Ngl.open_wks('png','TRANS_plot')
res=Ngl.Resources()
res.nglFrame=False
res.lbOrientation='horizontal'
res.sfXArray=lon
res.sfYArray=lat
res.mpFillOn=True
res.mpOceanFillColor='Transparent'
res.mpLandFillColor='Gray90'
res.mpInlandWaterFillColor='Gray90'
plot=Ngl.contour_map(wks,var,res)
txres=Ngl.Resources()
txres.txFontHeightF=0.014
Ngl.text_ndc(wks,f.variables['hgt'].attributes['long_name'],\
0.20,0.78,txres)
Ngl.text_ndc(wks,f.variables['hgt'].attributes['units'],\
0.95,0.78,txres)
Ngl.frame(wks)
Ngl.end()
'''
'''
#文本文件的读取
fname="/home/tangyuheng/data_ncl/shco12.txt"
f1name="shco12_c.txt"
with open(fname,'r') as f:
	shlines=f.readlines()

print(shlines)
#shline1s=shlines.split('\n')
#for shline1 in shline1s:
#	print(shline1)
#print('****')
print('****'+'****'.join(shlines))
'''
'''
#数组字符的转型
import numpy as np
a=['3.4','2.1','-2.6']
#anum=np.zeros(len(a),'d')
#for i in range(len(a)):
#	anum[i]=float(a[i])
#或者
anum=np.array(a).astype('d')
print(anum)

T=[273.4,265.5,277.7,285.5]
outputstr=['\n']*len(T)
for i in range(len(T)):
	outputstr[i]=str(T[i])+outputstr[i]
fileout=open('one_col_temp.txt','w')
fileout.writelines(outputstr)
fileout.close()
filein=open('one_col_temp.txt','r')
inputstr=filein.readlines()
filein.close()
Tnew=np.array(inputstr).astype('d')
print(inputstr)
'''
'''
#NumPy基础：数组与向量化计算
import numpy as np
data=np.random.randn(2,3)
data1=[[1,2.3,4,5],[1,2,3,4]]
arr1=np.array(data1)
print(arr1.dtype)
arr2=np.empty((2,3))
print(np.arange(15).dtype)
arr1=np.array([1,2,3],np.float64)
print(arr1.dtype)
float32_arr1=arr1.astype(np.float32)
print(float32_arr1.dtype)
numeric_strings=np.array(['1.25','2','-3.48'],dtype=np.string_)
print(numeric_strings.astype(float))
arr=np.array([[1.,2,3],[4.,5,6]])
print((arr**2>arr).dtype)

arr=np.arange(10)
print(arr[5:8])#5 6 7
arr[5:8]=12 #对切片的改变会体现在原数组上
print(arr)
c=arr[5:8].copy()

print(c,arr[5:8])
arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[2],arr2d[2][1])
arr3d=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr3d.shape)
print(arr3d[0],'\n\n',arr3d[0,:,:])
old_values=arr3d[0].copy()
arr3d[0]=43
print(arr3d)
arr3d[0]=old_values.copy()
print(arr3d)
print(arr2d[:2],arr2d[1,:2])
names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data=np.random.randn(7,4)
print(names=='Bob','\n',data[(names=='Bob')|(names=='Will')])
data[data<0]=0
print(data)
#神奇索引
arr=np.empty((8,4))
for i in range(8):
	arr[i]=i
print(arr)
print(arr[[4,3,0,6]])
print(arr[[-3,-5,-7]])
arr=np.arange(32).reshape((8,4))
print(arr)
print(arr[[1,5,7],[0,3,1]])
print(arr[[1,5,7,2]][:,[0,3,1,2]])
print(arr.T)
print("***************")
arr=np.arange(24).reshape((2,3,4))
print(arr)
print(arr.transpose(1,2,0))
print(arr.shape)
'''
'''
#通用函数：快速的逐元素数组函数
arr=np.arange(10)
print(np.sqrt(arr))
x=np.random.randn(8)
y=np.random.randn(8)
print(x,y)
np.maximum(x,y)#比较x和y各个元素，选出最小的元素
arr=np.random.randn(7)*5
print(arr)
remainder,whole_part=np.modf(arr)
print(remainder,whole_part)

#使用数组进行面向数组编程
points=np.arange(-5,5,0.01)#1000 equally spaced points
xs,ys=np.meshgrid(points,points)
print(ys)
z=np.sqrt(xs**2+ys**2)
print(z)
import matplotlib.pyplot as plt
plt.imshow(z,cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt(x^2+y^2) for a grid of values")
plt.show()
'''
'''
#将条件逻辑作为数组操作
xarr=np.array([1.1,1.2,1.3,1.4,1.5])
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,True,False])
result=np.where(cond,xarr,yarr)
print(result)
arr=np.random.randn(3,4)
print(arr)
print(np.where(arr>0,1,0))
print(arr.sum())
print(arr.mean())
print(arr.mean(axis=0))
arr=np.array([0,1,2,3,4,5,6,7])
print(arr.cumsum())
arr=np.random.randn(100)
print((arr>0).sum())
bools=np.array([False,False,True,False])
print(bools.any())#检查是否至少有一个True
print(bools.all())#检查是否每个都是True
#排序
arr=np.random.randn(6)
print(arr)
arr.sort()
print(arr)
#计算数组分位数
large_arr=np.random.randn(1000)
large_arr.sort()
print(large_arr[int(0.05*len(large_arr))])
#线性代数
x=np.array([[1.,2.,3.],[4.,5.,6.]])
y=np.array([[6.,23.],[-1,7],[8,9]])
print(x.dot(y))

#伪随机数
np.random.seed(1234)
rng=np.random.RandomState(1234)
print(rng.randn(10))
#example
import random
position=0
walk=[position]
steps=1000
for i in range(steps):
	step=1 if random.randint(0,1) else -1
	position+=step
	walk.append(position)#000000
plt.plot(walk[:100])
plt.show()
'''
#pandas入门
#pd.series
'''
obj=pd.Series([4,7,5,-3])
print(obj)
obj2=pd.Series([4,7,5,-3],index=['a','b','c','d'])
print(obj2.index)
print(obj2['c'])
print(obj2[obj2>0])
print(obj2*2)
print(np.exp(obj2))
print('b' in obj2)
sdata={'Ohio':35000,'Texas':71000,'Oregon':16000}
obj3=pd.Series(sdata)
print(obj3)
states=['California','Texas','Ohio']
obj4=pd.Series(sdata,index=states)
print(obj4)
print(pd.isnull(obj4))#检查是否缺失数据
print(pd.notnull(obj4))
print(obj3+obj4)
obj4.name='population'
obj4.index.name='state'
print(obj4)
'''
#DataFrame
'''
data={'state':['Ohio','Ohio','Ohio','Nevada','Nevada','Nevada'],
'year':[2000,2001,2002,2001,2002,2003],
'pop':[1.5,1.7,3.6,2.4,2.9,3.2]}
frame=pd.DataFrame(data)
print(frame)
print()
print(pd.DataFrame(data,columns=['year','state','pop']))
print(pd.DataFrame(data,columns=['year','state','pop','debt'],index=['one','two','three','four','five','six']))
frame2=pd.DataFrame(data,columns=['year','state','pop','debt'],index=['one','two','three','four','five','six'])
print(frame.columns)
print(frame['year'])
print(frame2.loc['two'])
#frame2['debt']=np.arange(6.)
#print(frame2)
val=pd.Series([-1.2,-1.5,-1.7],index=['two','four','five'])
frame2['debt']=val
print(frame2)
frame2['eastern']=frame2.state=='Ohio'
print(frame2)
del frame2['eastern']
print(frame2.columns)
pop={'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3=pd.DataFrame(pop)
print(frame3)
print(frame3.T)
print(pd.DataFrame(pop,index=[2001,2002,2003]))
pdata={'Onio':frame3['Ohio'][:-1],'Nevada':frame3['Nevada'][:2]}
print(pd.DataFrame(pdata))
frame3.index.name='year'
frame3.columns.name='state'
print(frame3)
print(frame3.values)
print(frame2.values)
'''
'''
#索引对象
obj=pd.Series(range(3),index=['a','b','c'])
index=obj.index
print(index)  #索引对象是不可变的，因此用户无法修改索引对象index[1]='d'
labels=pd.Index(np.arange(3))
print(labels)
obj2=pd.Series([1.5,-2.5,0],index=labels)
print(obj2)
print(obj2.index is labels)
pop={'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
frame3=pd.DataFrame(pop)
print('Ohio' in frame3.columns)
print(2003 in frame3.index)
#pandas索引对象可以包含重复的标签
'''
#重建索引
'''
obj=pd.Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
print(obj)
obj2=obj.reindex(['a','b','c','d','e'])
print(obj2)
obj3=pd.Series(['blue','purple','yellow'],index=[0,2,4])
print(obj3)
print(obj3.reindex(range(6),method='ffill'))
frame=pd.DataFrame(np.arange(9).reshape((3,3)),index=['a','c','d'],  \
columns=['Ohio','Texas','California'])
print(frame)
states=['Texas','Utah','California']
print(frame.reindex(columns=states))
#更简洁的方式
print(frame.loc[['a','b','c','d'],states])
'''
#轴向上删除条目
'''
obj=pd.Series(np.arange(5.),index=['a','b','c','d','e'])
print(obj)
new_obj=obj.drop('c')
print(new_obj)
print(obj.drop(['d','c']))
data=pd.DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado'
,'Utah','New York'],columns=['one','two','three','four'])
print(data)
print(data.drop(['Colorado','Ohio']))
print(data.drop('two',axis='columns'))
print(data.drop('Ohio',inplace=True))
print(data)
'''
#索引、选择与过滤Series的切片包括尾部
'''
obj=pd.Series(np.arange(5.),index=['a','b','c','d','e'])
print(obj['b':'c'])
data=pd.DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado'
,'Utah','New York'],columns=['one','two','three','four'])
print(data)
print(data[['two','one']])
print(data[:2])
print(data[data['three']>5])
data[data<5]=0
print(data)
#使用loc和iloc选择数据,loc轴标签，iloc整数标签
print(data.loc['Colorado',['two','three']])
print(data.iloc[2,[3,0,1]])
print(data.iloc[[1,2],[3,0,1]])
#整数索引
ser=pd.Series(np.arange(3.))
print(ser.loc[:1])
print(ser.iloc[:1])
#算术和数据对齐
s1=pd.Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
s2=pd.Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])
print(s1+s2)
df1=pd.DataFrame(np.arange(9.).reshape((3,3)),columns=list('bcd'),index=
['Ohio','Texas','Colorado'])
df2=pd.DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),index
=['Utah','Ohio','Texas','Oregon'])
print(df1+df2)
#使用填充值的算术方法
df1=pd.DataFrame(np.arange(12.).reshape((3,4)),columns=list('abcd'))
df2=pd.DataFrame(np.arange(20.).reshape((4,5)),columns=list('abcde'))
df2.loc[1,'b']=np.nan
print(df1)
print(df2)
print(df1+df2)
print(df1.add(df2,fill_value=0))
print(1/df1)
print(df1.reindex(columns=df2.columns,fill_value=0))
'''
#DataFrame和Series间的操作
'''
arr=np.arange(12.).reshape((3,4))
print(arr-arr[0])
frame=pd.DataFrame(np.arange(12.).reshape((4,3)),columns=list('bde'),
index=['Utah','Ohio','Texas','Oregon'])
series=frame.iloc[0]
print(frame)
print(series)
print(frame-series)
frame=pd.DataFrame(np.random.randn(4,3),columns=list('bed'),index=
['Utah','Ohio','Texas','Oregon'])
print(frame)
f=lambda x: x.max()-x.min()
print(frame.apply(f))
print(frame.apply(f,axis='columns'))
format=lambda x: '%.2f' %x
print(frame.applymap(format))
'''
#排序和排名
'''
print(frame.sort_index())
print(frame.sort_index(axis=1,ascending=False))
print(frame.sort_values(by='b'))
obj=pd.Series([7,-5,7,4,2,0,4])
print(obj.rank())
print(obj.rank(method='first'))
frame=pd.DataFrame({'b':[4.3,7,-3,2],'a':[0,1,0,1]})
print(frame)
print(frame.rank(axis='columns'))#一行的排序
'''
#含有重复标签的轴索引
'''
obj=pd.Series(range(5),index=['a','a','b','b','c'])
print(obj)
print(obj.index.is_unique)#判断标签是否唯一
print(obj['a'])
'''
#描述性统计的概述与计算
'''
df=pd.DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],
index=['a','b','c','d'],columns=['one','two'])
print(df)
print(df.sum())#同一列相加
print(df.sum(axis='columns',skipna=False))#遇到nan就输出nan,排除缺失值
print(df.idxmax())#返回最大值的索引值
print(df.describe())#输出多个汇总统计
'''
#相关性和协方差
'''
import pandas_datareader.data as web
all_data={ticker:web.get_data_yahoo(ticker) for ticker in ['AAPL',
'IBM','MSFT','GOOG']}
price=pd.DataFrame({ticker:data['Adj Close'] for ticker, data in all_data.items()})
volume=pd.DataFrame({ticker:data['Volume'] for ticker, data in all_data.items()})
returns=price.pct_change()
print(returns.tail())#转不出来
'''
#唯一值，计数和成员属性
'''
obj=pd.Series(['c','a','d','a','a','b','b','c','c'])
uniques=obj.unique()
print(uniques)
print(obj.value_counts())
print(pd.value_counts(obj.values,sort=False))
mask=obj.isin(['b','c'])
print(mask)
print(obj[mask])
to_match=pd.Series(['c','a','b','b','c','a'])
unique_vals=pd.Series(['c','b','a'])
print(pd.Index(unique_vals).get_indexer(to_match))
data=pd.DataFrame({'Qu1':[1,3,4,3,4],'Qu2':[2,3,1,2,3],'Qu3':[1,5,2,4,4]})
print(data)
result=data.apply(pd.value_counts)#value_counts是将重复值计数，并显示每个值在每列出现的次数
print(result)
'''
#文本格式数据的读写
'''
PATH='/home/tangyh/data_ncl/ex1.csv'
df=pd.read_csv('/home/tangyh/data_ncl/ex1.csv')
print(df)
print(pd.read_table('/home/tangyh/data_ncl/ex1.csv',sep=','))

print(pd.read_csv('/home/tangyh/data_ncl/ex1.csv',header=None))
print(pd.read_csv('/home/tangyh/data_ncl/ex1.csv',names=['a','b','c','d','message']))
names=['a','b','c','d','message']
print(pd.read_csv(PATH,names=names,index_col='message'))

PATH='/home/tangyh/data_ncl/csv_mindex.csv'
parsed=pd.read_csv(PATH,index_col=['key1','key2'])
print(parsed)

print(pd.read_table('/home/tangyh/data_ncl/ex1.csv',sep=',',skiprows=[1,3]))

result=pd.read_csv('/home/tangyh/data_ncl/ex5.csv',na_values=['NULL'])
print(result)
print(pd.isnull(result))
sentinels={'message':['foo','NA'],'something':['two']}
print(pd.read_csv('/home/tangyh/data_ncl/ex5.csv',na_values=sentinels))
'''
#将数据写入文本格式
'''
import sys
pd.options.display.max_rows=10
data=pd.read_csv('/home/tangyh/data_ncl/ex5.csv')
data.to_csv('/home/tangyh/data_ncl/out.csv',sep='|')
data.to_csv(sys.stdout,sep='|',na_rep='NULL',index=False,header=False)
dates=pd.date_range('1/1/2000',periods=7)
ts=pd.Series(np.arange(7),index=dates)
ts.to_csv(sys.stdout)
'''
#使用分隔格式
'''
import csv
with open('/home/tangyh/data_ncl/ex7.csv') as f:
	lines=list(csv.reader(f))
	reader=csv.reader(f,delimiter='|')
print(reader)
print(lines)
header,values=lines[0],lines[1:]
data_dict={h:v for h,v in zip(header,zip(*values))} 
print(data_dict)
class my_dialect(csv.Dialect):
	lineterminator='\n'
	delimiter=';'
	quotechar='"'
	quoting=csv.QUOTE_MINIMAL
with open('/home/tangyh/data_ncl/mydata.csv','w') as f:
	writer=csv.writer(f,dialect=my_dialect)
	writer.writerow(('one','two','three'))
	writer.writerow(('1','2','3'))
	writer.writerow(('4','5','6'))
	writer.writerow(('7','8','9'))
'''
#json数据
'''
obj="""
{"name":"Wes",
"places_lived":["United States","Spain","Germany"],
"pet":null,
"siblings":["name":"Scott","age":30,"pets":["Zeus","Zuko"]},
{"name":"Katie","age":38,
"pets":["Sixes","Stache","Cisco"]}]
}
"""
import json
result=json.loads(obj)
print(result)
'''

#XML和HTML：网络抓取
'''
tables=pd.read_html('examples/fdic_failed_bank_list.html')
print(len(tables))
'''
#二进制格式
#HDF5格式
'''
frame=pd.DataFrame({'a':np.random.randn(100)})
#store=pd.HDFStore('mydata.h5')
#store['obj1']=frame
#store['obj1_col']=frame['a']
#print(store)
#store.put('obj2',frame,format='table') #将frame放在h5文件
#print(store.select('obj2',where=['index>=10 and index<=15']))
frame.to_hdf('mydata.h5','obj3',format='table')
print(pd.read_hdf('mydata.h5','obj3',where=['index<5']))
'''
#读取和写入Microsoft Excel文件
'''
xlsx=pd.ExcelFile('/home/tangyh/data_ncl/englishB.xlsx')
frame=pd.read_excel(xlsx,'Sheet1')
print(frame)
writer=pd.ExcelWriter('/home/tangyh/data_ncl/englishout.xlsx')
frame.to_excel(writer,'Sheet1')
writer.save()
'''
#与Web API交互
'''
import requests
url='https://api.github.com/repos/pandas-dev/pandas/issues'
resp=requests.get(url)
print(resp)
data=resp.json()
print(data[0]['title'])
issues=pd.DataFrame(data,columns=['number','title','labels','state'])
print(issues)
'''
#与数据库交互
'''
import sqlite3
query="""
CREATE TABLE test
(a VARCHAR(20),b VARCHAR(20),c REAL,d INTEGER );"""
con=sqlite3.connect('mydata3.sqlite')
#print(con.execute(query))
#con.commit()

#data=[('Atlanta','Georgia',1.25,6),('Tallahassee','Florida',2.6,3),
#('Sacramento','California',1.7,5)]
#stmt="INSERT INTO test VALUES(?,?,?,?)"
#print(con.executemany(stmt,data))
#con.commit()
cursor=con.execute('select*from test')
rows=cursor.fetchall()
print(rows)
print(cursor.description)
print(pd.DataFrame(rows,columns=[x[0] for x in cursor.description]))

import sqlalchemy as sqla
db=sqla.create_engine('sqlite:///mydata3.sqlite')
print(pd.read_sql('select * from test',db))
'''
#处理缺失值
string_data=pd.Series(['aardvark','artichoke',np.nan,'avocado'])
print(string_data)
print(string_data.isnull())
string_data[0]=None
print(string_data.isnull())
#过滤缺失值
from numpy import nan as NA
data=pd.Series([1,NA,3.5,NA,7])
print(data.dropna())#丢弃缺失值
print(data[data.notnull()])
data=pd.DataFrame([[1,2,3],[4,NA,5],[6,NA,NA]])
cleaned=data.dropna()
print(cleaned)













