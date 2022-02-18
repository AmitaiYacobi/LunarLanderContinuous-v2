
import sys
import os

#--------------------------------discritization------------------------------------#
main_engine_values = [0, 0.5, 1]
sec_engine_values = [-1, -0.75, 0, 0.75, 1]
discrete_actions = [(x, y) for x in main_engine_values for y in sec_engine_values]
#----------------------------------------------------------------------------------#