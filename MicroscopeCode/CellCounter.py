'''Koen Bresters 2020
For a large part a copy of Microscope.py   

Some code that, when run, creates a pygame display that shows what the raspberry pi camera is currently seeing.
 - close the window to stop the program
 - press 'f' to save a photo of what is currently on screen
 - press 'p' to pause the video feed
 - Click on the screen to mark a cell
 - 
'''

import pygame
from FastCam import *
import pygame
from math import ceil
import time
import numpy as np
import json
import os

# draw array encoding an image on surface
# slow! use pygame.surfarray.blit_array instead
def blit_array(array, surface, x=0, y=0):
    display.blit(pygame.surfarray.make_surface(array), (x, y))

# save an array encoding an image as an image
def save_array(array, file_name):
    pygame.image.save(pygame.surfarray.make_surface(array), file_name)

# a function used to round the resolution to an allowed resolution
def roundup(x, base):
    return int(ceil(x / float(base))) * base

# a function that draws text to screen
def draw_text(pos, text, color, font):
    display.blit(font.render(text, False, color), pos)

# dictionary with resolutions for each camera mode
# see https://picamera.readthedocs.io/en/release-1.12/fov.html#camera-modes for source
camera_resolutions = {
    1: (1920, 1080),
    2: (2592, 1944),
    3: (2592, 1944),
    4: (1296, 972),
    5: (1296, 730),
    6: (640, 480),
    7: (640, 480)
}

# open .json-file with camera settings
settings_folder = "MicroscopeSettings"

# find all settings files
settings_options = []
for filename in os.listdir(settings_folder):
    if filename.endswith(".json"):
        settings_options.append(filename)

# present choice
for i, item in enumerate(settings_options):
    print("option {}: {}".format(i, item))
choice_result = input("Which settings would you like:\n")

# attempt to convert choice to the right file location
try:
    settings_location = settings_folder + "/" + settings_options[int(choice_result)]
except IndexError:
    print("This number is not an option")
    pygame.quit()
    quit()
except:
    print("Not an integer!")
    pygame.quit()
    quit()

# open settings file
settings_file = open(settings_location, "r")
settings = json.loads(settings_file.read()) # turn text into a dictionary
settings_file.close()

# put camera parameters into variables
sensor_mode = settings["sensor_mode"]
resolution = camera_resolutions[sensor_mode]
rounded_resolution = (roundup(resolution[0], 32), roundup(resolution[1], 16))
display_resolution = tuple(settings["display_resolution"])
framerate = settings["framerate"]

# screenshot paramaters
screenshot_file_name = "Screenshots/Screenshot"
screenshot_file_extension = ".jpg"

# pause parameters
pause_framerate = 60

# cell marking parameters
mark_color = (255, 0, 0) # rgb for red
mark_half_width = 2

# pygame initialisation
pygame.init()
display = pygame.display.set_mode(display_resolution)

# text parameters
text_color = (255, 0, 0) # rgb for red
text_font = pygame.font.SysFont("Arial", 25)
text_offset_x = 10
text_offset_y = 10
# calculate the size of one digit
len_digit = text_font.render("0", False, (0, 0, 0)).get_width()

class MyRecord(ContinuedRecord):
    def my_initialize(self):
        self.initialize(resolution, rounded_resolution, framerate, sensor_mode)
        self.doing_stuff = False
        self.stop = False
        self.marked_cells = []
        self.is_paused = False
        self.last_np_y_data = None

    def do_continue(self):
        if self.stop:
            self.stop = False
            return False
        else:
            return True

    def draw_frame(self, np_y_data):
        display.fill((255, 255, 255))
        pressed_f = False

        # event handling
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT: # when window is closed
                self.stop = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f: # when f is pressed
                    pressed_f = True
                elif event.key == pygame.K_r: # when r is pressed
                    self.marked_cells = []
                elif event.key == pygame.K_p: # when p is pressed
                    self.is_paused = not self.is_paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.marked_cells.append(pygame.mouse.get_pos())

        # put image on screen
        blit_array(np_y_data, display)

        # display marked cells
        for cell in self.marked_cells:
            rect = [cell[0]-mark_half_width, cell[1]-mark_half_width,
                    2*mark_half_width, 2*mark_half_width]
            pygame.draw.rect(display, mark_color, rect)

        # draw number of cells
        text = str(len(self.marked_cells))
        draw_text((display_resolution[0]-len_digit*len(text)-text_offset_x, text_offset_y),
                  text, text_color, text_font)

        # make screenshot
        if pressed_f:
            file_name = screenshot_file_name + time.strftime("%Y%m%d-%H%M%S") + screenshot_file_extension
            pygame.image.save(display, file_name)
            # draw the red box
            pygame.draw.rect(display, (255, 0, 0),
                             [0, 0, display_resolution[0], display_resolution[1]], 5)
        pygame.display.update()
        

    def on_pause(self):
        clock = pygame.time.Clock()
        self.draw_frame(self.last_np_y_data)
        clock.tick(pause_framerate)

    def on_record(self, y_data):        
        # make sure this method isnt being run on another thread
        if self.doing_stuff:
            print("frame dropped!")
            return
        self.doing_stuff = True
        
        
        # convert y_data to pygame and flip x,y axes
        np_y_data = np.array(y_data)
        np_y_data = np.transpose(np_y_data, (1, 0, 2))

        # crop to display_size
        x_min = (resolution[0]-display_resolution[0]) // 2
        y_min = (resolution[1]-display_resolution[1]) // 2
        x_max = (resolution[0]+display_resolution[0]) // 2
        y_max = (resolution[1]+display_resolution[1]) // 2
        np_y_data = np_y_data[x_min:x_max, y_min:y_max, :]

        # store np_y_data for when the frame is paused
        self.last_np_y_data = np_y_data
        
        # draw a frame
        self.draw_frame(np_y_data)

        # store that the method is no longer being ran
        self.doing_stuff = False

myRecord = MyRecord()
myRecord.my_initialize()
myRecord.run()
pygame.quit()
















    
