from math import sqrt, pow, sin, cos, acos, pi, sqrt
from random import random
import utils

#----------------------------------------------
# Math
#----------------------------------------------

def random_angle_curried(limit):
    def random_angle():
        non_uniform_norm = pow(abs(limit), 3)
        value = 0
        while value == 0 or random() < pow(abs(value), 3) / non_uniform_norm:
            value = random_range(-limit, +limit)
        return value
    return random_angle

def subtract_points(point1, point2):
    return {
        'x': point1['x'] - point2['x'],
        'y': point1['y'] - point2['y']
    }

def cross_product(point1, point2):
    return point1['x'] * point2['y'] - point1['y'] * point2['x']

def vector_length_squared(v):
    return v['x'] * v['x'] + v['y'] * v['y']

def vector_length(v):
    return sqrt(vector_length_squared(v))

def angle_between(v1, v2):
    angle_rad = acos((v1['x'] * v2['x'] + v1['y'] * v2['y']) / (vector_length(v1) * vector_length(v2)))
    angle_deg = angle_rad * 180 / pi
    return angle_deg

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def length(point1, point2):
    vector = subtract_points(point2, point1)
    return vector_length(vector)

def sin_degrees(deg):
    return sin(deg * pi / 180)

def cos_degrees(deg):
    return cos(deg * pi / 180)

def do_line_segment_intersect(p, p2, q, q2, omitEnds):
    r = subtract_points(p2, p)
    s = subtract_points(q2, q)
    uNumerator = cross_product(subtract_points(q, p), r)
    denominator = cross_product(r, s)
    if uNumerator == 0 and denominator == 0:
        return False
    if denominator == 0:
        return False
    u = uNumerator / denominator
    t = cross_product(subtract_points(q, p), s) / denominator
    doSegmentsIntersect = None
    if not omitEnds:
        doSegmentsIntersect = t >= 0 and t <= 1 and u >= 0 and u <= 1
    else:
        doSegmentsIntersect = t > 0.001 and t < 1-0.001 and u > 0.001 and u < 1-0.001
    if doSegmentsIntersect:
        return {
            'x': p['x'] + t * r['x'],
            'y': p['y'] + t * r['y'],
            't': t
        }
    return doSegmentsIntersect

def random_range(min, max):
    return random() * (max - min) + min

def min_degree_difference(d1, d2):
    diff = abs(d1 - d2) % 180
    return min(diff, abs(diff - 180))

def length_squared(point1, point2):
    vector = subtract_points(point2, point1)
    return vector_length_squared(vector)

def equal_vector(v1, v2):
    diff = subtract_points(v1, v2)
    length = vector_length_squared(diff)
    return length < 0.00000001

def dot_product(point1, point2):
    return point1['x'] * point2['x'] + point1['y'] * point2['y']

def multiply_vector_scalar(v, n):
    return {
        'x': v['x'] * n,
        'y': v['y'] * n
    }

def project(v, onto):
    _dot_product = dot_product(v, onto)
    return {
        'dot_product': _dot_product,
        'projected': multiply_vector_scalar(onto, _dot_product / vector_length_squared(onto))
    }

def add_points(point1, point2):
    return {
        'x': point1['x'] + point2['x'],
        'y': point1['y'] + point2['y']
    }

def distance_to_line(P, A, B):
    AP = subtract_points(P, A)
    AB = subtract_points(B, A)
    result = project(AP, AB)
    AD = result['projected']
    D = add_points(A, AD)
    return {
      'distance2': length_squared(D, P),
      'pointOnLine': D,
      'lineProj2': sign(result['dot_product']) * vector_length_squared(AD),
      'length2': vector_length_squared(AB)
    }
