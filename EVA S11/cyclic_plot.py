# Plot a graph between LR and iterations
import math
import matplotlib.pyplot as plt

lr_range= []
def triangular_plot(iterations, stepsize, lr_max, lr_min ):
  

	for i in range(iterations):
	  cycle= math.floor(1+ (i/(2*stepsize)))
	  x= abs((i/stepsize) - (2* (cycle)) + 1) 
	  lr_t= lr_max + (lr_min - lr_max )*x
	  lr_range.append(lr_t)


	plt.plot(range(iterations), lr_range )