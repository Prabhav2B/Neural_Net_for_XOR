#Backpropogation implementaion program for XOR using numpy
import numpy as np 
import time

#Backpropogation Neural Network with 1 hidden layer

#Activation Function used: Sigmoid
def act(Squash_this):
  return 1/(1 + np.exp(-Squash_this))

#Derivative of activation Function
def act_prime(g):
  return g*(1 - g)

#Set Learning Rate Here
a = 3
#Seeding(Ensures same random numbers)
np.random.seed(1)

#Training set
X = np.array([[0,0],[0,1],[1,0],[1,1]])                     
Y = np.array([[0],[1],[1],[0]])
Iter = 6000

#Random Initialization of Weights between -.5 to .5
W1 = 2*np.random.random((2,4)) - 1 
W2 = 2*np.random.random((4,1)) - 1  

#Clock Starts
t_start = time.clock()


for no in range(Iter):

  #Forward Pass

  hidden = act(np.dot(X,W1))
  output = act(np.dot(hidden,W2))
  err_out = Y-output

  if (no%1000) == 0:
    print "Error : " + str(np.mean(np.abs(err_out)))

  #Back-Propogation

  #The Error between the predicted and actual output.
  del_out = err_out*act_prime(output)

  err_hid = np.dot(del_out, W2.T)
  del_hid = err_hid * act_prime(hidden)
  
  #Parameter Updation
  W2 += a*np.dot(hidden.T,del_out)
  W1 += a*np.dot(X.T, del_hid)

#Time Stops here
t_end = time.clock()
print('Elapsed time ', t_end - t_start)

print "Final Result: "
print output
