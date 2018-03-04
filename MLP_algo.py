import numpy as np
import math

y=[]
x=[]
E=[]
alpha=0.01
Total_error=0.0
file=open("input.txt")
for line in file:
	x.append(list(map(float,line.split('=')[0].split(" "))))	
	y.append(list(map(float,line.strip().split("=")[1])))

y=np.asarray(y)
levels=np.unique(y)
u=np.array([[1.0 for i in range(len(x))]])
x=np.asarray(x)
x=np.append(u.T,x,axis=1)
w1=np.array([[1.0 for i in range(len(x[0]))]for i in range(len(x[0]))]) #w1 [[no of input nodes ]no of hidden nodes-1]
w2=np.array([[1.0 for i in range(len(w1)+1)]for i in range(len(levels))]) #w2 [[no of hidden nodes +1]no of output nodes] 
e=np.zeros([len(x),len(levels)],dtype=float) #error array
h=np.zeros([len(x),len(w1)],dtype=float)
h=np.append(u.T,h,axis=1)
O=np.zeros([len(x),len(levels)],dtype=float)
sub=np.zeros([len(x),len(levels)],dtype=float)

itr=0
while(itr<=1000):
	for m in range(len(x)):
		# Forward Pass
		v=np.zeros([1,len(w1)],dtype=float)
		for i in range(len(x[0])):
			v[0][i]=x[m].dot(w1[i].T)
			h[m][i+1]=1/(1+np.exp(-v[0][i]))

		# Hidden to output 
		Y=np.zeros([len(x),len(levels)],dtype=float)
		for i in range(len(levels)):
			Y[m][i]=h[m].dot(w2[i].T)
			O[m][i]=1/(1+np.exp(-Y[0][i]))

		# Backward Pass
		# Local error in output layer
		del2=np.zeros([1,len(levels)],dtype=float)
		for i in range(len(levels)):
			del2[0][i]=y[i]-O[m][i]*(1-O[m][i])

		del1=np.zeros([1,len(w1)],dtype=float)
		for i in range(1,len(w2)):
			p=0
			for j in range(len(levels)):
				p=p+del2[0][i]*w2[j][i]
			del1[0][i]=p
		for i in range(len(levels)):
			e[m][i]=y[i]-O[m][i] # m is the iteration no of sample

		for i in range(len(levels)):
			E.append(1/2*(math.pow(e[m][i],2)))


		# Weight Updation
		# Input to hidden layer
		for i in range(len(w1)):
			for j in range(len(x[0])):
				w1[i][j]=w1[i][j]+alpha*del1[0][i]*x[m][i] # m is the iteration no of sample
		# Hidden to output
		for i in range(len(w2)):
			for j in range(len(w1)+1):
				w2[i][j]=w2[i][j]+alpha*del2[0][i]*h[m][i] # m is the iteration no of sample

		Total_error+=E[m]
	itr=itr+1;
print(w1)
print(w2)

p=input("enter sample of features equal to input.txt :")
p=p.split(" ")
p=[1]+p

p=[float(x) for x in p]
sam=[x for x in p]
sam=np.asarray(sam)
v2=np.zeros([1,len(sam)],dtype=float)
h2=np.zeros([1,len(w2[0])],dtype=float)
Y2=np.zeros([1,len(levels)],dtype=float)
O2=np.zeros([1,len(levels)],dtype=float)
for i in range(len(sam)):
	v2[0][i]=sam.dot(w1[i].T)
	h2[0][i]=1/(1+np.exp(-v[0][i]))

for i in range(len(levels)):
	Y2[0][i]=h2[len(h2)-1].dot(w2[i].T)
	O2[0][i]=1/(1+np.exp(-Y2[0][i]))

index=0
for i in range(len(x)):
	if(np.array_equal(sam,x[i])):
		index=i

calculated=np.argmax(O2[0],axis=0)

O2[0][calculated]=1.0
for i in range(len(O2)):
	if i!=calculated:
		O2[0][i]=0.0
print(O2)