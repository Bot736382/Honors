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
        self.x = x #COM for the object
        self.y = y #COM for the object (does not include the bots)
        self.theta = theta
        self.u = 0 #Assume originally stationary
        self.v = 0 #Assume originally stationary
    
    def update_state(self, dt, Bot1, Bot2):
        u1 = Bot1.u
        v1 = Bot1.v
        x1 = Bot1.x
        y1 = Bot1.y

        x2 = Bot2.x
        u2 = Bot2.u
        v2 = Bot2.v        
        y2 = Bot2.y

        x_original= self.x
        y_original= self.y
        theta_original = self.theta

        #In case of debugging, these equations might be the best place to start
        bot_dist = math.sqrt((x2-x1)^2 + (y2-y1)^2) 
        omega = ((x2-x1)*(v2-v1) - (y2-y1)*(u2-u1)) / (bot_dist)^2
        self.u = u1 + omega*(y1-y_original)
        self.v = v1 - omega*(x1-x_original)

        self.x += self.u *dt
        self.y += self.v *dt
        self.theta += omega *dt
        
class Bot:
    def __init__(self,x,y, max_speed):
        self.x = x
        self.y = y
        self.max_speed = max_speed
        self.heading_angle = 0
        self.u = 0 # initialise to 0
        self.v = 0 # initialise to 0    


##################################################################################
# 2. Define the destination point
##################################################################################
class Destination:
    def __init__(self, x, y):
        self.x = x
        self.y = y


##################################################################################
# 3. MPC Constraints?
##################################################################################


