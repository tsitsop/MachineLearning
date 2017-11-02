import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math

def createData():
	x1 = random.randint(0,1)
	x2 = random.randint(0,1)

	return np.array([[x1,x1**2], 
					 [x2,x2**2]])

# number of runs
n = 1

# Data set consists of two points, where 
#  first element of each is x, second is x^2
d = np.array([[0,0]*2])



while n > 0:


	n-= 1