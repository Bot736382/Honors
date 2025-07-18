import math
import matplotlib.pyplot as plt
# import matplotlib
import numpy as np

##################################################################################
# 1. Define the box as well as robots as classes 
# This could be potentially helpful when increasing the number of bots later on
##################################################################################
class box:
    def __init__(self,x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

class Bot:
    def __init__(self,x,y, u_max, v_max):
        self.x = x
        self.y = y
        self.u_max = u_max
        self.v_max = v_max

##################################################################################
# Define the destination point
##################################################################################
class Destination:
    def __init__(self, x, y):
        self.x = x
        self.y = y
