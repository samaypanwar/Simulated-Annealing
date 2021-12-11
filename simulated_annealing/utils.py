# from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def plot_function(func, r_min = -3, r_max=3, optima=(0, 0 , 0)):

    # sample input range uniformly at 0.1 increments
    xaxis = arange(r_min, r_max, 0.2)
    yaxis = arange(r_min, r_max, 0.2)
    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    vector = [x, y]
    results = func(vector)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='winter')
    axis.scatter(*optima, color='r', ls='--')
    # show the plot
    pyplot.show()

# def animate(x, y):
#     fig = plt.figure()
#     axis = plt.axes(xlim =(-50, 50),
#                     ylim =(-50, 50))

#     line, = axis.plot([], [], lw = 2)

#     def init():
#         line.set_data([], [])
#         return line,


# surface plot for 2d objective function