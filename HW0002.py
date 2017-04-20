#HW0002
#1. generate five random numbers and print out
#2. generate 10, 100, 1000, 10000, 100000 random numbers and caculate thier means , standard deviations 
import numpy as np
five = np.random.rand(5)
print("The five random numbers are: ")
for j in five :
    print (j)

n = [10,100,1000,10000,100000]
# n=10
for i in n :
    print("The mean of %d numbers are:"%i ,np.mean(2*np.random.sample(i)-1))
    print("The standard deviation of %d numbers are:"%i ,np.std(2*np.random.sample(i)-1))
