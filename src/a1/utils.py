import random

def clamp(x, a, b):
    return min(max(x, a), b)

def flip(p):
   return random.random() < p