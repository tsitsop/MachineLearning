import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import quad

def createData():
	x1 = random.uniform(-1,1)
	x2 = random.uniform(-1,1)

	# Data set consists of two points, where 
	#  first element of each is some x, second is x^2
	return np.array([[x1,x1**2], 
					 [x2,x2**2]])

def getSlope(d):
	try:
		return (d[1][1]-d[0][1])/(d[1][0]-d[0][0])
	except ZeroDivisionError:
		# line is vertical
		return None
	
def getYIntercept(d, m):
	if (m != None):
		return d[0][1] - m*d[0][0]
	else:
		return None

def getBias(gbar, x):
	return (gbar[0]*x+gbar[1]-x**2)**2

# number of runs
n = 1000

# going to have n data sets
d = [np.array([0,0])]*n

# going to have n g's, where a g is a slope and y intercept
g = [(0.,0.)]*n

# represents x space
xspace = np.arange(-1., 1., 0.05)

# repeat experiment n times
for i in range(n):
	d[i] = createData()

	m = getSlope(d[i])
	b = getYIntercept(d[i], m)

	g[i] = (m, b)
	# for visual purposes,  plot the line
	plt.plot(xspace, m*xspace+b, "r")

# find some average line gbar
avgM = float(sum([x[0] for x in g]))/n
avgB = float(sum([x[1] for x in g]))/n
gbar = (avgM, avgB)

# # find eout
# eout_k = [0.]*n
# for i in range(n):
# 	# find mean squared error of g_k(x) and f(x) for each data set k
# 	#			   ___m___*____x_____+___b___	  ____f(x)____
# 	eout_k[i] = ( (g[i][0]*d[i][0][0]+g[i][1]) - d[i][0][0]**2 )**2

# # find the average mean squared error with respect to the n data sets
# eout_x = sum(eout_k)/n
















# biasx = lambda x: (gbar[0]*x+gbar[1]-x**2)**2 

# avgM2 = float(sum([x[0]**2 for x in g]))/n
# avgB2 = float(sum([x[1]**2 for x in g]))/n
# avgG2 = (avgM2, avgB2)

# diff = (avgG2[0] - gbar[0]**2, avgG2[1] - gbar[1]**2)

# varx = lambda x: (diff[0]*x**2 + 2*diff[0]*diff[1]*x + diff[1])
# var = quad(varx, -1, 1)
# print var

# # varx = lambda
# bias = quad(biasx, -1, 1)
# print bias




# Plot gbar against all the lines,f(x), show plot
plt.plot(xspace, gbar[0]*xspace+gbar[1], "b")
plt.plot(xspace, xspace**2, "g")
plt.title('2.24')
plt.axis([-1, 1, -1, 1])
plt.xlabel('x')
plt.ylabel('x^2')
plt.show()