import numpy as np

a=np.array([10,30,20,30,40,50,40,30,20,10])
a=a/50
b=a-np.mean(a)
c=b/np.std(a)
print(np.mean(a))
print(b)
print(c)