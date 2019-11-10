import numpy as np
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
#两数组对应元素相乘:方法一
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














