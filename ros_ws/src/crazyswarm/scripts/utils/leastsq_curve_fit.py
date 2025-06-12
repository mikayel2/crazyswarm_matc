# The following source is used to build this class https://python4mpia.github.io/fitting_data/least-squares-fitting.html
import numpy
import scipy.optimize as optimization
import matplotlib.pyplot as plt

class LSQ():
    def __init__(self,t_data, gamma_data):
        # Constructor
        self.t_data = t_data
        self.gamma_data = gamma_data


    def func(self,x, a, b):
        return a + b*x

    def optz(self):
        # Placeholder for method2
        ab , var = optimization.curve_fit(self.func, self.t_data, self.gamma_data)
        return ab[0] , ab[1]


if __name__ == "__main__":
    # xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
    # ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])

    xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
    ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])    

    lsq = LSQ(xdata, ydata)
    a, b = lsq.optz()
    print(a,b)
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(xdata, ydata, label='Data')

    # Plot the fitted curve
    x = numpy.linspace(min(xdata), max(xdata), 1000)
    y = a + b * x
    plt.plot(x, y, 'r', label='Fitted curve')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least Squares Fit')
    plt.legend()
    plt.show()



    # Generate artificial data = straight line with a=0 and b=1
    # plus some noise.

    # xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
    # ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])
    # # Initial guess.
    
    # def func(x, a, b):
    #     return a + b*x
    
    # print(optimization.curve_fit(func, xdata, ydata))
