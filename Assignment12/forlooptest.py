import time

l = range(100)

t=time.time()
q=0
for i in l:
    q+=1
print(time.time()-t)


t=time.time()
q=0
[q+1 for i in l]
print(time.time()-t)