import math
import matplotlib.pyplot as plt
# import matplotlib
import matplotlib.patches as patches
import numpy as np
from DrawShapes import sketch


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
        self.u = u1 + omega*(y1-y_original)
        self.v = v1 - omega*(x1-x_original)

        self.x += self.u *dt
        self.y += self.v *dt

        Bot1.x += self.u *dt
        Bot2.x += self.u *dt

        Bot1.y += self.v *dt
        Bot2.y += self.v *dt
        self.theta += omega *dt
        print(f"Box Position: ({self.x}, {self.y})")

    def rotate(self, angle):
        pass

    def translate(self):
        pass

    def combined_rotate_translate(self):
        pass
        
class Bot:
    def __init__(self,x,y, length, width,max_speed):
        self.x = x
        self.y = y

        self.length=length
        self.width=width

        self.max_speed = max_speed
        self.heading_angle = 0
        self.u = 0 # initialise to 0
        self.v = 0 # initialise to 0   
        self.path_x = []
        self.path_y = [] 

        self.bot_x = 0
        self.bot_y = 0

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

def dist_f_error(Bot, Destination):
    """
    Calculate the distance from the bot to the destination.
    """
    return math.sqrt((Bot.x - Destination.x)**2 + (Bot.y - Destination.y)**2)

##################################################################################
# 3. Describe the path
##################################################################################
path_x = []
path_y = []

def path_add(Box, Destination, Bot1, Bot2, dt):
    """
    This function describes the path of the box as it moves towards the destination.
    It updates the position of the box and the bots based on their velocities.
    """
    Box.update_state(dt, Bot1, Bot2)

    # Store the path for plotting
    path_x.append(Box.x)
    path_y.append(Box.y)
    Bot1.path_x.append(Bot1.x)
    Bot1.path_y.append(Bot1.y)
    Bot2.path_x.append(Bot2.x)
    Bot2.path_y.append(Bot2.y)

##################################################################################
# 4. Open loop Code
##################################################################################

### Declaring the box and the bots
box1 = box(0, 0, 0)
Bot1 = Bot(2, -2, 3,1, 0.5)
Bot2 = Bot(-2, 2,3,1, 0.5)

Dest= Destination(10, 10)
# Time step for simulation
dt = 0.1

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

while dist_f_error(box1, Dest) > 0.1:
    
    sketch.print_plot(ax, box1, Bot1, Bot2, Dest)
    
    heading_angle = math.atan2(Dest.y - box1.y, Dest.x - box1.x)
    Bot1.u = Bot1.max_speed * math.cos(heading_angle)
    Bot1.v = Bot1.max_speed * math.sin(heading_angle)
    Bot1.heading_angle = math.degrees(heading_angle)
    Bot2.heading_angle = math.degrees(heading_angle)  
    Bot2.u = Bot1.u
    Bot2.v = Bot1.v

    path_add(box1, Dest, Bot1, Bot2, dt)

    
    plt.pause(0.05)  # Pause to update the figure

plt.ioff()  # Turn off interactive mode
plt.show()  # Show final frame


# plot path_x and path_y
plt.figure()
plt.plot(path_x, path_y, 'r-', label='Path of the Box')
plt.plot(Dest.x, Dest.y, 'yo', label='Destination')
plt.plot(Bot1.path_x, Bot1.path_y, 'b--', label='Path of Bot 1')
plt.plot(Bot2.path_x, Bot2.path_y, 'g--', label='Path of Bot 2')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Initial Positions')
plt.legend()
plt.grid()
plt.show()

# sketch.print_shape()  # Call the function to print the shape description
