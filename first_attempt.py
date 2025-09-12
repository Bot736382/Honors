import math
import matplotlib.pyplot as plt
import numpy as np
from DrawShapes import sketch
from Definitions import box, Bot, Gripper, Destination

# Initialise the path lists
path_x = []
path_y = []

def dist_f_error(Bot, Destination):
    ## Calculate the distance from the bot to the destination.
    return math.sqrt((Bot.x - Destination.x)**2 + (Bot.y - Destination.y)**2)

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

### Declare the grippers
gripper1 = Gripper()
gripper2 = Gripper()

### Declaring the box and the bots
box1 = box(0, 0, 0)
Bot1 = Bot(2, -2, 0.2, 0.2, 0.2, gripper1, 1, 45)
Bot2 = Bot(-2, 2,0.2, 0.2, 0.2, gripper2, 1, 225)

Dest= Destination(1.5, 1.9)

# Time step for simulation
dt = 0.1

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

while dist_f_error(box1, Dest) >= 0.1:
    
    sketch.print_plot(ax, box1, Bot1, Bot2, Dest)
    
    heading_angle = math.atan2(Dest.y - box1.y, Dest.x - box1.x)
    Bot1.u = Bot1.max_speed * math.cos(heading_angle)
    Bot1.v = Bot1.max_speed * math.sin(heading_angle)
    Bot1.heading_angle = math.degrees(heading_angle)
    Bot2.heading_angle = math.degrees(heading_angle)  
    Bot2.u = Bot1.u
    Bot2.v = Bot1.v

    path_add(box1, Dest, Bot1, Bot2, dt)
    
    plt.pause(0.02)  # Pause to update the figure

plt.ioff()  # Turn off interactive mode
plt.show()  # Show final frame


# plot path_x and path_y
plt.figure()
plt.plot(path_x, path_y, 'r-', label="Path of the Object's COM")
plt.plot(Dest.x, Dest.y, 'yo', label='Destination')
plt.plot(Bot1.path_x, Bot1.path_y, 'b--', label='Path of Bot 1')
plt.plot(Bot2.path_x, Bot2.path_y, 'g--', label='Path of Bot 2')
# plt.xlim(-20, 20)
# plt.ylim(-20, 20)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Initial Positions')
plt.legend()
plt.grid()
plt.show()

# sketch.print_shape()  # Call the function to print the shape description
