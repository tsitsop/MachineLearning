# DOESNT WORK CORRECTLY



import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

def formula(x):
	return ((-w[1]/w[2]) * x - (w[0]/w[2]))


##### initialize random inputs, making sure linearly seperable and 2d #####
#####  also generate ys for the inputs #####
def generateRandomInputs(n, r): # number of inputs, range
	x1 = []
	x2 = []

	i=0
	while i < n:
		testx1 = random.randint(0,r-1)
		testx2 = random.randint(0,r-1)
		
		# make sure not on line
		if testx1 != testx2:
			i += 1
			x1.append(testx1)
			x2.append(testx2)

	##### initialize y list by seeing where point lies compared to line #####
	y = []
	for j in range(0,n):
		if (x1[j] > x2[j]):
			y.append(-1)
		elif (x1[j] < x2[j]):
			y.append(1)

	return [x1,x2,y]

##### find correct classifications for each data point #####
def generateClassifications(x1,x2,y,n):
	pos = [] 
	neg = [] 

	for j in range(0,n):
		if y[j] == 1:
			pos.append((x1[j],x2[j])) # create list of positive points
		else:
			neg.append((x1[j],x2[j])) # create list of negative points
	
	return [pos, neg]

##### plot all of the x and y points
def plotPoints(pos, neg):
	xpos = [x[0] for x in pos]
	ypos = [x[1] for x in pos]
	xneg = [x[0] for x in neg]
	yneg = [x[1] for x in neg]
	plt.plot(xpos, ypos, 'bo', label='Positive')
	plt.plot(xneg, yneg, 'rx', label='Negative')

# get all misclassified points
def getMisclassified(x1, x2, y, w, n):
	misclassifiedPoints = []

	# go through all points and make list of misclassified points
	for i in range(0, n):
		# get weighted sum -- w0x0 + w1x1 + w2x2
		currentSum =  w[0] + (x1[i]*w[1]) + (x2[i]*w[2])
		# if weighted sum has different sign than y, point is misclassified
		if ((currentSum > 0 and y[i] < 0) or (currentSum < 0 and y[i] > 0) or (currentSum == 0)):
			misclassifiedPoints.append((x1[i], x2[i], i))
	
	return misclassifiedPoints

# def plotLine(w):
	# TODO

## run PLA
 # x1 is first dimension of input
 # x2 is second dimension of input
 # y is correct classification of input
 # pos is the points that should have positive y
 # neg is the points that should have negative y
 # n is the number of input points
 ##
def pla(x1, x2, y, w, pos, neg, n):
	t=1
	while True:
		all_misclassified = getMisclassified(x1, x2, y, w, n)

		# if no points misclassified, PLA worked.
		if (len(all_misclassified) == 0):
			poop = np.array(range(0, 10))
			y = formula(poop)

			plt.plot(poop,y, label='final hypothesis')
			break

		#otherwise, pick a random misclassified point and improve w by it
		num = random.randint(0, len(all_misclassified)-1)
		misclassified = all_misclassified[num]

		# update weights
		w[0] += y[misclassified[2]]
		w[1] += y[misclassified[2]] * misclassified[0]
		w[2] += y[misclassified[2]] * misclassified[1]
		
		t += 1



n = 10
##### generate initial random inputs and their y values #####
g = generateRandomInputs(n,10)
x1 = g[0]
x2 = g[1]
y = g[2]

##### separate points into lists of positive, negative classifications #####
c = generateClassifications(x1,x2,y, n)
pos = c[0]
neg = c[1]

##### initialize weights #####
w = [0, 0, 0]

# ##### Plot all points, coloring them if correct or not #####
plotPoints(pos, neg)

# ##### plot the correct line #####
x = np.array(range(-1,11))  
plt.plot(x, x, label='target function')  

pla(x1,x2,y,w,pos,neg,n)



plt.title('part a')
plt.axis([-1, 10, -1, 13])
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper right')


plt.show()  




# # have two dimensions of inputs - need to define a target function f that will
# # represent the line that separates the data
# # target function is x