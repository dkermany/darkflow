import numpy as np

"""
Boxes defined as a dict with keys
{xn, yn, xx, yx}
"""

# Intersection over Union using coordinates
def IOU(A, B):
  return intersection(A, B) / \
         float(union(A, B))

# Return Area if overlap
def intersection(A, B):
  assert A['xx'] > A['xn'] and B['xx'] > B['xn'] and \
         A['yx'] > A['yn'] and B['yx'] > B['yn']
  w = min(A['xx'], B['xx']) - max(A['xn'], B['xn'])
  h = min(A['yx'], B['yx']) - max(A['yn'], B['yn'])
  if w >= 0 and h >= 0:
    return w * h
  return 0

def union(A, B):
  i = intersection(A, B)
  return area(A) + area(B) - i

def area(A):
  area = (A['xx'] - A['xn']) * (A['yx'] - A['yn'])  
  assert area >= 0
  return area
