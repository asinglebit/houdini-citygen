from math import floor, fmod, sqrt, pow, sin, cos, radians, acos, pi, sqrt
from random import randint, seed, random
from types import MethodType
import copy

#----------------------------------------------
# Math
#----------------------------------------------

class SimplexNoise:
    _GRAD3 = ((1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),(1,1,0),(0,-1,1),(-1,1,0),(0,-1,-1))
    _F2 = 0.5 * (sqrt(3.0) - 1.0)
    _G2 = (3.0 - sqrt(3.0)) / 6.0
    permutation = (151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,9,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180)
    period = len(permutation)
    permutation = permutation * 2
    randint_function = randint
    def __init__(self, period=None, permutation_table=None, randint_function=None):
        if randint_function is not None:  # do this before calling randomize()
            if not hasattr(randint_function, '__call__'):
                raise TypeError('randint_function has to be a function')
            self.randint_function = randint_function
            if period is None:
                period = self.period  # enforce actually calling randomize()
        if period is not None and permutation_table is not None:
            raise ValueError('Can specify either period or permutation_table, not both')
        if period is not None:
            self.randomize(period)
        elif permutation_table is not None:
            self.permutation = tuple(permutation_table) * 2
            self.period = len(permutation_table)
    def randomize(self, period=None):
        if period is not None:
            self.period = period
        perm = list(range(self.period))
        perm_right = self.period - 1
        for i in list(perm):
            j = self.randint_function(0, perm_right)
            perm[i], perm[j] = perm[j], perm[i]
        self.permutation = tuple(perm) * 2
    def noise2(self, x, y):
        s = (x + y) * self._F2
        i = floor(x + s)
        j = floor(y + s)
        t = (i + j) * self._G2
        x0 = x - (i - t)
        y0 = y - (j - t)
        if x0 > y0:
            i1 = 1; j1 = 0
        else:
            i1 = 0; j1 = 1
        x1 = x0 - i1 + self._G2
        y1 = y0 - j1 + self._G2
        x2 = x0 + self._G2 * 2.0 - 1.0
        y2 = y0 + self._G2 * 2.0 - 1.0
        perm = self.permutation
        ii = int(i) % self.period
        jj = int(j) % self.period
        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1]] % 12
        gi2 = perm[ii + 1 + perm[jj + 1]] % 12
        tt = 0.5 - x0**2 - y0**2
        if tt > 0:
            g = self._GRAD3[gi0]
            noise = tt**4 * (g[0] * x0 + g[1] * y0)
        else:
            noise = 0.0
        tt = 0.5 - x1**2 - y1**2
        if tt > 0:
            g = self._GRAD3[gi1]
            noise += tt**4 * (g[0] * x1 + g[1] * y1)
        tt = 0.5 - x2**2 - y2**2
        if tt > 0:
            g = self._GRAD3[gi2]
            noise += tt**4 * (g[0] * x2 + g[1] * y2)
        return noise * 70.0

def random_angle_curried(limit):
    def random_angle():
        non_uniform_norm = pow(abs(limit), 3)
        value = 0
        while value == 0 or random() < pow(abs(value), 3) / non_uniform_norm:
            value = math_random_range(-limit, +limit)
        return value
    return random_angle

def math_subtract_points(point1, point2):
    return {
        'x': point1['x'] - point2['x'],
        'y': point1['y'] - point2['y']
    }

def math_cross_product(point1, point2):
    return point1['x'] * point2['y'] - point1['y'] * point2['x']

def math_vector_length_squared(v):
    return v['x'] * v['x'] + v['y'] * v['y']

def math_vector_length(v):
    return sqrt(math_vector_length_squared(v))

def math_angle_between(v1, v2):
    angle_rad = acos((v1['x'] * v2['x'] + v1['y'] * v2['y']) / (math_vector_length(v1) * math_vector_length(v2)))
    angle_deg = angle_rad * 180 / pi
    return angle_deg

def math_sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def math_length(point1, point2):
    vector = math_subtract_points(point2, point1)
    return math_vector_length(vector)

def math_sin_degrees(deg):
    return sin(deg * pi / 180)

def math_cos_degrees(deg):
    return cos(deg * pi / 180)

def math_do_line_segment_intersect(p, p2, q, q2, omitEnds):
    r = math_subtract_points(p2, p)
    s = math_subtract_points(q2, q)
    uNumerator = math_cross_product(math_subtract_points(q, p), r)
    denominator = math_cross_product(r, s)
    if uNumerator == 0 and denominator == 0:
        return False
    if denominator == 0:
        return False
    u = uNumerator / denominator
    t = math_cross_product(math_subtract_points(q, p), s) / denominator
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

def math_random_range(min, max):
    return random() * (max - min) + min

def math_min_degree_difference(d1, d2):
    diff = abs(d1 - d2) % 180
    return min(diff, abs(diff - 180))

def math_length_squared(point1, point2):
    vector = math_subtract_points(point2, point1)
    return math_vector_length_squared(vector)

def math_equal_vector(v1, v2):
    diff = math_subtract_points(v1, v2)
    length = math_vector_length_squared(diff)
    return length < 0.00000001

def math_dot_product(point1, point2):
    return point1['x'] * point2['x'] + point1['y'] * point2['y']

def math_multiply_vector_scalar(v, n):
    return {
        'x': v['x'] * n,
        'y': v['y'] * n
    }

def math_project(v, onto):
    dot_product = math_dot_product(v, onto)
    return {
        'dot_product': dot_product,
        'projected': math_multiply_vector_scalar(onto, dot_product / math_vector_length_squared(onto))
    }

def math_add_points(point1, point2):
    return {
        'x': point1['x'] + point2['x'],
        'y': point1['y'] + point2['y']
    }

def math_distance_to_line(P, A, B):
    AP = math_subtract_points(P, A)
    AB = math_subtract_points(B, A)
    result = math_project(AP, AB)
    AD = result['projected']
    D = math_add_points(A, AD)
    return {
      'distance2': math_length_squared(D, P),
      'pointOnLine': D,
      'lineProj2': math_sign(result['dot_product']) * math_vector_length_squared(AD),
      'length2': math_vector_length_squared(AB)
    }

#----------------------------------------------
# Globals
#----------------------------------------------

noise = SimplexNoise()

#----------------------------------------------
# Constants
#----------------------------------------------

SEED = 1
SEGMENT_COUNT_LIMIT = 2000
ROAD_SNAP_DISTANCE = 3
MINIMUM_INTERSECTION_DEVIATION = 10
HIGHWAY_SEGMENT_LENGTH = 40
HIGHWAY_BRANCH_PROBABILITY = 0.1
HIGHWAY_BRANCH_POPULATION_THRESHOLD = 0.1
DEFAULT_SEGMENT_LENGTH = 30
DEFAULT_BRANCH_PROBABILITY = 0.2
NORMAL_BRANCH_POPULATION_THRESHOLD = 0.1
NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY = 5
RANDOM_BRANCH_ANGLE = random_angle_curried(3)
RANDOM_STRAIGHT_ANGLE = random_angle_curried(5)

#----------------------------------------------
# Population map
#----------------------------------------------

def heatmap_pop_at(x, y):
    value1 = (noise.noise2(x/10000, y/10000) + 1) / 2
    value2 = (noise.noise2(x/20000 + 500, y/20000 + 500) + 1) / 2
    value3 = (noise.noise2(x/20000 + 1000, y/20000 + 1000) + 1) / 2
    return pow((value1 * value2 + value3) / 2, 2)

def heatmap_pop_on_road(start, end):
    return (heatmap_pop_at(start['x'], start['y']) + heatmap_pop_at(end['x'], end['y'])) / 2

#----------------------------------------------
# Generator
#----------------------------------------------

class Segment:
    collider = None
    road_revision = 0
    dir_revision = None
    length_revision = None
    cached_dir = None
    cached_length = None
    start = None
    end = None
    t = 0
    q = None
    links = None

    def __init__(self, start, end, t, q):
        self.t = t or 0
        self.q = copy.deepcopy(q) if q else {}
        self.start = copy.deepcopy(start)
        self.end = copy.deepcopy(end)
        self.collider = {
            'x': min(self.start['x'], self.end['x']),
            'y': min(self.start['y'], self.end['y']),
            'width': abs(self.start['x'] - self.end['x']),
            'height': abs(self.start['y'] - self.end['y'])
        }
        self.links = {
            'b': [],
            'f': []
        }

    def set_start(self, start):
        self.start = start
        self.road_revision += 1

    def set_end(self, end):
        self.end = end
        self.road_revision += 1

    def dir(self):
        if self.dir_revision != self.road_revision:
            self.dir_revision = self.road_revision
            vector = math_subtract_points(self.end, self.start)
            self.cached_dir = -1 * math_sign(math_cross_product({'x': 0, 'y': 1}, vector)) * math_angle_between({'x': 0, 'y': 1}, vector)
        return self.cached_dir

    def length(self):
        if self.length_revision != self.road_revision:
            self.length_revision = self.road_revision
            self.cached_length = math_length(self.start, self.end)
        return self.cached_length

    def links_for_end_containing(self, segment):
        if segment in self.links['b']:
            return self.links['b']
        elif segment in self.links['f']:
            return self.links['f']
        else:
            return None

    def start_is_backwards(self):
        if len(self.links['b']) > 0:
            return math_equal_vector(self.links['b'][0].start, self.start) or math_equal_vector(self.links['b'][0].end, self.start)
        else:
            return math_equal_vector(self.links['f'][0].start, self.end) or math_equal_vector(self.links['f'][0].end, self.end)

    def split(self, point, segment, segment_list):
        start_is_backwards = self.start_is_backwards()
        split_part = segment_from_existing(self)
        segment_list.append(split_part)
        split_part.set_end(point)
        self.set_start(point)
        split_part.links['b'] = list(self.links['b'])
        split_part.links['f'] = list(self.links['f'])

        first_split = None
        second_split = None
        fix_links = None

        if start_is_backwards:
            first_split = split_part
            second_split = self
            fix_links = split_part.links['b']
        else:
            first_split = self
            second_split = split_part
            fix_links = split_part.links['f']

        for link in fix_links:
            if self in link.links['b']:
                index = link.links['b'].index(self)
                link.links['b'][index] = split_part
            elif self in link.links['f']:
                index = link.links['f'].index(self)
                link.links['f'][index] = split_part

        first_split.links['f'] = []
        first_split.links['f'].append(segment)
        first_split.links['f'].append(second_split)
        second_split.links['b'] = []
        second_split.links['b'].append(segment)
        second_split.links['b'].append(first_split)
        segment.links['f'].append(first_split)
        segment.links['f'].append(second_split)

def segment_from_existing(segment, t = None, start = None, end = None, q = None):
    return Segment(
        start if start else segment.start,
        end if end else segment.end,
        t or segment.t,
        q if q else segment.q
    )

def segment_from_direction(start, dir, length, t, q):
    _dir = dir or 90
    _length = length or DEFAULT_SEGMENT_LENGTH
    _end = {
        'x': start['x'] + length * math_sin_degrees(_dir),
        'y': start['y'] + length * math_cos_degrees(_dir)
    }
    return Segment(start, _end, t, q)

def global_goals(previous_segment):
    new_branches = []
    if not previous_segment.q.has_key('severed') or not previous_segment.q['severed']:
        def template(direction, length, t, q = {}):
            return segment_from_direction(previous_segment.end, direction, length, t, q)
        continue_straight = template(previous_segment.dir(), previous_segment.length(), 0, previous_segment.q)
        straight_pop = heatmap_pop_on_road(continue_straight.start, continue_straight.end)
        if previous_segment.q.has_key('highway') and previous_segment.q['highway']:
            random_straight = template(previous_segment.dir() + RANDOM_STRAIGHT_ANGLE(), previous_segment.length(), 0, previous_segment.q)
            random_pop = heatmap_pop_on_road(random_straight.start, random_straight.end)
            road_pop = None
            if random_pop > straight_pop:
                new_branches.append(random_straight)
                road_pop = random_pop
            else:
                new_branches.append(continue_straight)
                road_pop = straight_pop
            if road_pop > HIGHWAY_BRANCH_POPULATION_THRESHOLD:
                if random() < HIGHWAY_BRANCH_PROBABILITY:
                    left_highway_branch = template(previous_segment.dir() - 90 + RANDOM_BRANCH_ANGLE(), previous_segment.length(), 0, previous_segment.q)
                    new_branches.append(left_highway_branch)
                elif random() < HIGHWAY_BRANCH_PROBABILITY:
                    right_highway_branch = template(previous_segment.dir() + 90 + RANDOM_BRANCH_ANGLE(), previous_segment.length(), 0, previous_segment.q)
                    new_branches.append(right_highway_branch)
        elif straight_pop > NORMAL_BRANCH_POPULATION_THRESHOLD:
            new_branches.append(continue_straight)
        
        if straight_pop > NORMAL_BRANCH_POPULATION_THRESHOLD:
            if random() < DEFAULT_BRANCH_PROBABILITY:
                left_branch = template(previous_segment.dir() - 90 + RANDOM_BRANCH_ANGLE(), DEFAULT_SEGMENT_LENGTH, NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY if previous_segment.q.has_key('highway') and previous_segment.q['highway'] else 0)
                new_branches.append(left_branch)
            elif random() < DEFAULT_BRANCH_PROBABILITY:
                right_branch = template(previous_segment.dir() + 90 + RANDOM_BRANCH_ANGLE(), DEFAULT_SEGMENT_LENGTH, NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY if previous_segment.q.has_key('highway') and previous_segment.q['highway'] else 0)
                new_branches.append(right_branch)

    for i in range(0, len(new_branches), 1):
        def do(branch = new_branches[i]):
            def setup_branch_links(self, previous = previous_segment):
                for link in previous.links['f']:
                    self.links['b'].append(link)
                    link.links_for_end_containing(previous).append(self)
                previous.links['f'].append(self)
                self.links['b'].append(previous)
            branch.setup_branch_links = MethodType(setup_branch_links, branch, Segment)
        do()
    return new_branches

def local_constraints(segment, segments):
    def a():
        return True
    priority = 0
    func = [a]
    q = {
        't': None
    }
    for i in range(0, len(segments), 1):
        other = segments[i]
        if priority <= 4:
            intersection = math_do_line_segment_intersect(segment.start, segment.end, other.start, other.end, True)
            if intersection:
                if q['t'] == None or intersection['t'] < q['t']:
                    q['t'] = intersection['t']
                    priority = 4
                    def temp_func_1(_other = other, _intersection = intersection):
                        if math_min_degree_difference(_other.dir(), segment.dir()) < MINIMUM_INTERSECTION_DEVIATION:
                            return False
                        _other.split(_intersection, segment, segments)
                        segment.end = _intersection
                        segment.q['severed'] = True
                        return True
                    func[0] = temp_func_1
        if priority <= 3:
            if math_length(segment.end, other.end) <= ROAD_SNAP_DISTANCE:
                point = other.end
                priority = 3
                def temp_func_2(_other = other, _point = point):
                    segment.end = _point
                    segment.q['severed'] = True
                    links = _other.links['f'] if _other.start_is_backwards() else _other.links['b']
                    are_there_duplicates = False
                    for link in links:
                        terminator_a = math_equal_vector(link.start, segment.end) and math_equal_vector(link.end, segment.start)
                        terminator_b = math_equal_vector(link.start, segment.end) and math_equal_vector(link.end, segment.start)
                        if terminator_a or terminator_b:
                            are_there_duplicates = True
                            break
                    if are_there_duplicates:
                        return False
                    for link in links:
                        link.links_for_end_containing(_other).append(segment)
                        segment.links['f'].append(link)
                    links.append(segment)
                    segment.links['f'].append(_other)
                    return True
                func[0] = temp_func_2
        if priority <= 2:
            distance_to_line = math_distance_to_line(segment.end, other.start, other.end)
            distance2 = distance_to_line['distance2']
            pointOnLine = distance_to_line['pointOnLine']
            lineProj2 = distance_to_line['lineProj2']
            length2 = distance_to_line['length2']
            if distance2 < ROAD_SNAP_DISTANCE * ROAD_SNAP_DISTANCE and lineProj2 >= 0 and lineProj2 <= length2:
                priority = 2
                def temp_func_3(_other = other, point = pointOnLine):
                    segment.end = point
                    segment.q['severed'] = True
                    if math_min_degree_difference(_other.dir(), segment.dir()) < MINIMUM_INTERSECTION_DEVIATION:
                        return False
                    _other.split(point, segment, segments)
                    return True
                func[0] = temp_func_3
    return func[0]()

def generate():

    # Initial setup

    seed(SEED)
    noise.randomize(SEED)
    priority_queue = []
    segments = []

    # Setup two opposing highway segments

    root_segment = Segment({'x': 0, 'y': 0}, {'x': HIGHWAY_SEGMENT_LENGTH, 'y': 0}, 0, {'highway': True})
    opposite_direction = segment_from_existing(root_segment)
    new_end = {
        'x': root_segment.start['x'] - HIGHWAY_SEGMENT_LENGTH,
        'y': opposite_direction.end['y']
    }
    opposite_direction.set_end(new_end)
    opposite_direction.links['b'].append(opposite_direction)

    # Initialize priority queue

    priority_queue.append(root_segment)
    priority_queue.append(opposite_direction)

    # Generate segments
    while len(priority_queue) > 0 and len(segments) < SEGMENT_COUNT_LIMIT:
        min_t = None
        min_t_i = 0
        for i, segment in enumerate(priority_queue):
            if min_t == None or segment.t < min_t:
                min_t = segment.t
                min_t_i = i
        min_segment = priority_queue.pop(min_t_i)
        accepted = local_constraints(min_segment, segments)
        if accepted == True:
            if hasattr(min_segment, 'setup_branch_links'):
                min_segment.setup_branch_links()
            segments.append(min_segment)
            for new_segment in global_goals(min_segment):
                new_segment.t = min_segment.t + 1 + new_segment.t
                priority_queue.append(new_segment)

    return segments

#----------------------------------------------
# Houdini
#----------------------------------------------

segments = generate()

node = hou.pwd()
geo = node.geometry()
for segment in segments:
    poly = geo.createPolygon(False)
    start_point = geo.createPoint()
    start_point.setPosition((segment.start['x'], 0, segment.start['y']))
    poly.addVertex(start_point)
    end_point = geo.createPoint()
    end_point.setPosition((segment.end['x'], 0, segment.end['y']))
    poly.addVertex(end_point)
    
