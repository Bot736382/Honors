import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math

def print_shape():
    print("This is a box shape.")

def print_bot(x,y,length, width, theta, ax):
    # print(f"Box Length: {length}, Width: {width}, Heading Angle: {theta}")
    phi = math.atan(width/length)
    k=math.radians(theta)
    diag = math.sqrt(length**2 + width**2)
    rect = patches.Rectangle((x-(0.5*diag*math.cos(k+phi)), y-(0.5*diag*math.sin(k+phi))), length, width, angle=theta,
                            linewidth=2, edgecolor='blue', facecolor='lightgray')

    ax.add_patch(rect)

def draw_arm(x_B, y_B, l, theta_1, theta_B,ax):
    line=mlines.Line2D([x_B, x_B + l*math.cos(math.radians(theta_B + theta_1))],
                       [y_B, y_B + l*math.sin(math.radians(theta_B + theta_1))], color='red', linewidth=2)
    ax.add_line(line)



def print_plot(ax, box1, Bot1, Bot2, Dest):
    ax.clear()  # Clear previous frame
    ax.plot(box1.x, box1.y, 'ro', label='Object')
    ax.plot(Bot1.x, Bot1.y, 'bo', label='Bot 1')
    ax.plot(Bot2.x, Bot2.y, 'go', label='Bot 2')
    ax.plot(Dest.x, Dest.y, 'yo', label='Destination')
    print_bot(Bot1.x,Bot1.y,Bot1.length,Bot1.width, Bot1.heading_angle, ax)
    print_bot(Bot2.x,Bot2.y,Bot2.length,Bot2.width, Bot2.heading_angle, ax)
    draw_arm(Bot1.x, Bot1.y, 1, Bot1.arm_angle, Bot1.heading_angle, ax)
    draw_arm(Bot2.x, Bot2.y, 1, Bot2.arm_angle, Bot2.heading_angle, ax)
    # ax.set_xlim(-5, 15)
    # ax.set_ylim(-5, 15)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Dynamic Positioning')
    ax.legend()
    ax.grid(True)