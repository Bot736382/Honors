import numpy as np
import math

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
        bot_dist = math.sqrt((x2-x1)**2 + (y2-y1)**2) 
        omega = ((x2-x1)*(v2-v1) - (y2-y1)*(u2-u1)) / (bot_dist)**2
        theta_original+= omega*dt
        # print(f"Box Orientation: {math.degrees(theta_original)} degrees")
        self.u = u1 + omega*(y1-y_original)
        self.v = v1 - omega*(x1-x_original)

        self.x += self.u *dt
        self.y += self.v *dt

        Bot1.x += self.u *dt
        Bot2.x += self.u *dt

        Bot1.y += self.v *dt
        Bot2.y += self.v *dt
        self.theta += omega *dt
        print(f"Object Centroid Position: ({self.x}, {self.y})")

    # def rotate(self, angle):
    #     pass

    # def translate(self):
    #     pass

    # def combined_rotate_translate(self):
    #     pass

class Gripper:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.u = 0
        self.v = 0
        self.omega = 0

class Bot:
    def __init__(self, x, y, length, width, max_speed, gripper: Gripper, arm_len, theta_1):
        # Set position
        self.x = x
        self.y = y

        # Set dimensions
        self.length=length
        self.width=width

        # Set motion constraints
        self.max_speed = max_speed
        self.orientation = 0
        self.heading_angle = 0
        self.u = 0 # initialise to 0
        self.v = 0 # initialise to 0   
        self.path_x = []
        self.path_y = [] 

        self.bot_x = 0
        self.bot_y = 0

        self.arm_length = arm_len  # Length of the arm
        self.arm_angle = theta_1  # Angle of the arm in degrees
        
        self.gripper = gripper
        self.gripper.x= x + arm_len * math.cos(math.radians(theta_1))
        print(f"Gripper x: {self.gripper.x}")
        self.gripper.y= y + arm_len * math.sin(math.radians(theta_1))
        print(f"Gripper y: {self.gripper.y}")
        self.theta_2 = 0
        

##################################################################################
# 2. Define the destination point
##################################################################################
class Destination:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Trajectory:
    def __init__(self, Destination):
        self.path_x = np.array([])
        self.path_y = np.array([])
        self.speed_x = np.array([])
        self.speed_y = np.array([])
        self.final_x = Destination.x
        self.final_y = Destination.y