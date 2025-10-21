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
        self.theta = theta # Angle pose of the object
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
        self.theta_2=0

class Bot:
    # The bot must have it's own x,y, length, width, v_Bx, v_By, theta_B, theta_1, arm_length, theta_2, gripper
    def __init__(self, x, y, length, width, max_speed, gripper: Gripper, arm_len, theta_1):
        # Set position
        self.x = x
        self.y = y
        self.q1 = math.radians(theta_1)  # Convert to radians
        self.q2 = arm_len # Length of the arm
        self.q3 = 0 # gripper angle

        # Set dimensions of the bot
        self.length=length
        self.width=width

        # Set motion constraints
        self.max_speed = max_speed
        self.orientation = 0
        self.heading_angle = 0
        self.u = 0 # initialise to 0
        self.v = 0 # initialise to 0   
        self.path_x = [] # WHILE Coding, adds the positions of the bot to a list
        self.path_y = [] # WHILE Coding, adds the positions of the bot to a list
        
        self.gripper = gripper # inherit the gripper class
        self.gripper.x= self.x + self.arm_length * math.cos(math.radians(self.arm_angle)) 
        self.gripper.y= self.y + self.arm_length * math.sin(math.radians(self.arm_angle))
        print(f"Gripper x: {self.gripper.x}")
        print(f"Gripper y: {self.gripper.y}")
        self.gripper.theta_2 = 0
        

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