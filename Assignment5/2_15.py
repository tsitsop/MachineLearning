import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math


n = 11

x=np.array([[0]*n]*n)

for i in range(0,n):
	for j in range(0, n):
		if ((i-(n-1)/2) + (j-(n-1)/2) == 0):
			x[i][j] = 0
		elif ((i-(n-1)/2) + (j-(n-1)/2) > 0):
			x[i][j] = 1
		else:
			x[i][j] = -1

bound = np.array(range(n))
for i in range(n):
	bound[i] = i-(n-1)/2

print "  ", bound
for i in range(n):
	if (i-(n-1)/2 >= 0):
		print i-(n-1)/2, "", x[i]
	else:
		print i-(n-1)/2, x[i]