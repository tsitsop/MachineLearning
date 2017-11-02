import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import time
import math

def flipCoin():
	return random.randint(0, 1) 


start_time = time.time()
# number of times to run the experiment
n = 10000
v_1s = [0.] * n #frequency of heads that the first coin of each trial got
v_rands = [0.] * n #frequency of heads that some random coin in each trial got
v_mins = [0.] * n #frequency of heads that the coin with smallest number of heads in each trial got

epsilons = list(range(6)) # the different possible epsilon values
for i in range(len(epsilons)): # divide by 10 to make them into the decimals
	epsilons[i] /= 10.

prob_part_c = [0.] * len(epsilons) #list that holds number of heads for each epsilon


for k in range(n):
	# list of 1000 coins
	# the value of each element is the number of heads a coin gets in 10 flips
	l = [0] * 1000

	for i in range(1000): #1000 coins
		for j in range(10): #10 flips per coin
			l[i] += flipCoin() #+1 if flip is a heads

	# the number of heads for coins
	c_1 = l[0] #number of heads the first coin flipped got
	c_rand = l[random.randint(0, 999)] #number of heads some random coin got

	# get cmin
	c_min = 11
	for i in range(1000):
		if (l[i] < c_min):
			c_min = l[i] #smallest number of heads any coin got

	v_1 = c_1/10.  # frequency of heads that the first coin got
	v_rand = c_rand/10. #frequency of heads some random coin got
	v_min = c_min/10. #frequency of heads the coin with smallest number of heads got

	v_1s[k] = v_1 
	v_rands[k] = v_rand
	v_mins[k] = v_min

	# do math for part c
	
	#for each value of epsilon, see if the difference is greater than epsilon>
	# if it is, then we can add 1 to prob_part_c
	for i in range(len(epsilons)): 
		if (abs(c_1 - 5) > (epsilons[i]*10)):
			prob_part_c[i] += 1 # this means the difference was greater than epsilons[i]


print("--- %s seconds ---" % (time.time() - start_time))


# Plot histograms on one figure

fig1, subplots = plt.subplots(3, 1, sharex="all", sharey="all")

subplots[0].hist(v_1s, facecolor="blue", label="nu_1")
subplots[0].set_title('Histogram of nu_1')
plt.xlabel('nu')
plt.ylabel('frequency of nu')
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

subplots[1].hist(v_rands, facecolor="red", label="nu_rand")
subplots[1].set_title('Histogram of nu_rand')

subplots[2].hist(v_mins, facecolor="green", label="nu_min")
subplots[2].set_title('Histogram of nu_min')

fig1.show()

# Plot estimates for probability and Hoeffding bound
fig2 = plt.figure()
hoeffding = [0.] * len(epsilons)
for i in range(len(epsilons)): # for each epsilon
	prob_part_c[i] /= n

	hoeffding[i] = 2 * math.e **(-2*(epsilons[i]**2)*10)

plt.plot(epsilons, prob_part_c, label="P(|nu-mu| > epsilon)")
plt.plot(epsilons, hoeffding, label="Hoeffding")
plt.xlabel("epsilon")
plt.ylabel("function")
plt.legend(loc='upper right')
fig2.show()
plt.show()  