#Backpropogation implementaion program for XOR using numpy
import numpy as np 

#Packpropogation Neural Network with 1 hidden layer
class nnXor:


	#Activation Function used: ReLU
	#PH = Place Holder
	def act(PH):
		return PH*(PH>0)

	#Derivative of activation Function
	def act_prime(PH):
		if PH>0:
			return 1
		else:
			return 0


	#Initialization
	def __init__(self,alpha,Input_X,Input_Y):
		self.a = alpha

		self.x = np.array(Input_X) # 4 X 2
		self.y = np.array(Input_X) # 4 X 1 

		#Seeding(Ensures same random numbers)
		np.random.seed(1)

		#Random Initialization of Weights between -.5 to .5
		self.W1 = np.random.random((4,5))-.5   # 4 X 5
		self.W2 = np.random.random((1,5))-.5	  # 1 X 5

		#Hidden Layer Initialization
		self.Hidden = np.zeros((4,1))  # 4 X 1
		self.output = np.zeros((4,1))  # 4 X 1

		#Delta Initialization
		self.Del_out = np.zeros((1,5))   # 1 X 5
		self.Del_hid = np.zeros((4,5))  # 4 X 5



	#Forward Pass
	def Forward_Pass(self):
		
		self.Hidden = self.act(np.dot(self.W1,self.x))  # 4 X( 5 * 4 )X 2

		self.output = self.act(np.dot(self.W2,self.Hidden)) # 1 X( 5 * 4 )X 2

	#Back-Propogation
	def Back_Prop(self):
		
		#The Error between the predicted and actual output.
		self.Del_out = self.output-self.y 
		self.Del_hid = np.dot(self.W2,Err)*act_prime(self.hidden)

		#Parameter Updation
		self.W2 = self.W2 - self.a*self.Del_out
		self.W1 = self.W1 - self.a*self.Del_hid

#Training set
X = np.array([(0,0),(0,1),(1,0),(1,1)])											
Y = np.array([0,1,1,0])
Iter = 6000

#Object Created
Exec = nnXor(.01,X,Y)

#Clock Starts
t_start = time.clock()

#Training Iterations
for no in range[Iter]:
	Exec.Forward_Pass()
	Exec.Back_Prop()

#Time Stops here
t_end = time.clock()
print('Elapsed time ', t_end - t_start)