import numpy as N
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
#read nc
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
import Nio
fname="/home/tangyuheng/data_ncl/shco12.txt"
f1name="shco12_c.txt"
with open(fname,'r') as f:
	shlines=f.readlines()

print(shlines)
#shline1s=shlines.split('\n')
#for shline1 in shline1s:
#	print(shline1)
print('\n'.join(shlines))














