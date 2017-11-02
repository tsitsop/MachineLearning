import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math

epsilons =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] 
print epsilons

c1 = 0

c2 = 0

v1 = 0.
v2 = 0.

for i in range(6):
	c1+= random.randint(0, 1) 
	c2+ random.randint(0, 1) 

v1 = c1/6.
v2 = c2/6.
r=[0.]*len(epsilons)

for i in range(len(epsilons)):
	if (abs(v1-.5) > abs(v2-.5)):
		r[i] = abs(v1-.5) - epsilons[i]
	else:
		r[i] = abs(v2-.5) - epsilons[i]




hoeffding = [0.] * len(epsilons)

for i in range(len(epsilons)):
	hoeffding[i] = 2*2*math.e**(-2*6*epsilons[i]**2)


print hoeffding

plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.plot(epsilons, r, label="R")

plt.plot(epsilons, hoeffding, label="Hoeffding")
plt.xlabel("epsilon")
plt.ylabel("function")
plt.legend(loc='upper right')
plt.show()