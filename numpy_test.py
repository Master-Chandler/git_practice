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
#计算位温
def theta(p,T,p0=1000.0,kappa=0.286):
	return T*(p0/p)**(kappa)
p=N.array([1020.0,1013.5,1017.2])
T=N.array([273.1,280.4,276.6])
print(theta(p,T))




