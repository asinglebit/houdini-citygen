from math import floor, fmod, sqrt, pow, sin, cos, radians, acos, pi, sqrt
from random import randint, seed, random
from types import MethodType
import copy

from noise import SimplexNoise
import utils

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

    @classmethod
    def segment_from_existing(cls, segment, t = None, start = None, end = None, q = None):
        return cls(
            start if start else segment.start,
            end if end else segment.end,
            t or segment.t,
            q if q else segment.q
        )

    @classmethod
    def segment_from_direction(cls, start, dir, length, t, q):
        _dir = dir or 90
        _length = length
        _end = {
            'x': start['x'] + length * utils.sin_degrees(_dir),
            'y': start['y'] + length * utils.cos_degrees(_dir)
        }
        return cls(start, _end, t, q)

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
            vector = utils.subtract_points(self.end, self.start)
            self.cached_dir = -1 * utils.sign(utils.cross_product({'x': 0, 'y': 1}, vector)) * utils.angle_between({'x': 0, 'y': 1}, vector)
        return self.cached_dir

    def length(self):
        if self.length_revision != self.road_revision:
            self.length_revision = self.road_revision
            self.cached_length = utils.length(self.start, self.end)
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
            return utils.equal_vector(self.links['b'][0].start, self.start) or utils.equal_vector(self.links['b'][0].end, self.start)
        else:
            return utils.equal_vector(self.links['f'][0].start, self.end) or utils.equal_vector(self.links['f'][0].end, self.end)

    def split(self, point, segment, segment_list):
        start_is_backwards = self.start_is_backwards()
        split_part = Segment.segment_from_existing(self)
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

#----------------------------------------------
# Globals
#----------------------------------------------

noise = SimplexNoise()

#----------------------------------------------
# Constants
#----------------------------------------------

# SEED = 1
# SEGMENT_COUNT_LIMIT = 2000
# ROAD_SNAP_DISTANCE = 3
# MINIMUM_INTERSECTION_DEVIATION = 10
# HIGHWAY_SEGMENT_LENGTH = 40
# HIGHWAY_BRANCH_PROBABILITY = 0.1
# HIGHWAY_BRANCH_POPULATION_THRESHOLD = 0.1
# DEFAULT_SEGMENT_LENGTH = 30
# DEFAULT_BRANCH_PROBABILITY = 0.2
# NORMAL_BRANCH_POPULATION_THRESHOLD = 0.1
# NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY = 5
# RANDOM_BRANCH_ANGLE = random_angle_curried(3)
# RANDOM_STRAIGHT_ANGLE = random_angle_curried(5)

SEED = hou.node(".").parm("seed").eval()
SEGMENT_COUNT_LIMIT = hou.node(".").parm("SEGMENT_COUNT_LIMIT").eval()
ROAD_SNAP_DISTANCE = hou.node(".").parm("ROAD_SNAP_DISTANCE").eval()
MINIMUM_INTERSECTION_DEVIATION = hou.node(".").parm("MINIMUM_INTERSECTION_DEVIATION").eval()
HIGHWAY_SEGMENT_LENGTH = hou.node(".").parm("HIGHWAY_SEGMENT_LENGTH").eval()
HIGHWAY_BRANCH_PROBABILITY = hou.node(".").parm("HIGHWAY_BRANCH_PROBABILITY").eval()
HIGHWAY_BRANCH_POPULATION_THRESHOLD = hou.node(".").parm("HIGHWAY_BRANCH_POPULATION_THRESHOLD").eval()
DEFAULT_SEGMENT_LENGTH = hou.node(".").parm("DEFAULT_SEGMENT_LENGTH").eval()
DEFAULT_BRANCH_PROBABILITY = hou.node(".").parm("DEFAULT_BRANCH_PROBABILITY").eval()
NORMAL_BRANCH_POPULATION_THRESHOLD = hou.node(".").parm("NORMAL_BRANCH_POPULATION_THRESHOLD").eval()
NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY = hou.node(".").parm("NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY").eval()
RANDOM_BRANCH_ANGLE = utils.random_angle_curried(hou.node(".").parm("RANDOM_BRANCH_ANGLE").eval())
RANDOM_STRAIGHT_ANGLE = utils.random_angle_curried(hou.node(".").parm("RANDOM_STRAIGHT_ANGLE").eval())

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

def global_goals(previous_segment):
    new_branches = []
    if not previous_segment.q.has_key('severed') or not previous_segment.q['severed']:
        def template(direction, length, t, q = {}):
            return Segment.segment_from_direction(previous_segment.end, direction, length or DEFAULT_SEGMENT_LENGTH, t, q)
        continue_straight = template(previous_segment.dir(), previous_segment.length(), 0, previous_segment.q)
        straight_pop = heatmap_pop_on_road(continue_straight.start, continue_straight.end)
        # If highway
        if previous_segment.q.has_key('highway') and previous_segment.q['highway']:
            random_straight = template(previous_segment.dir() + RANDOM_STRAIGHT_ANGLE(), previous_segment.length(), 0, previous_segment.q)
            random_pop = heatmap_pop_on_road(random_straight.start, random_straight.end)
            road_pop = None
            # Then continue in the direction of the highest population
            if random_pop > straight_pop:
                new_branches.append(random_straight)
                road_pop = random_pop
            else:
                new_branches.append(continue_straight)
                road_pop = straight_pop
            # And if higher than population threshold, also branch to left or right
            if road_pop > HIGHWAY_BRANCH_POPULATION_THRESHOLD:
                if random() < HIGHWAY_BRANCH_PROBABILITY:
                    left_highway_branch = template(previous_segment.dir() - 90 + RANDOM_BRANCH_ANGLE(), previous_segment.length(), 0, previous_segment.q)
                    new_branches.append(left_highway_branch)
                elif random() < HIGHWAY_BRANCH_PROBABILITY:
                    right_highway_branch = template(previous_segment.dir() + 90 + RANDOM_BRANCH_ANGLE(), previous_segment.length(), 0, previous_segment.q)
                    new_branches.append(right_highway_branch)
        # If street and higher than population threshold, continue straight
        elif straight_pop > NORMAL_BRANCH_POPULATION_THRESHOLD:
            new_branches.append(continue_straight)
        
        # If straight continuation higher than population threshold, branch
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
            intersection = utils.do_line_segment_intersect(segment.start, segment.end, other.start, other.end, True)
            if intersection:
                if q['t'] == None or intersection['t'] < q['t']:
                    q['t'] = intersection['t']
                    priority = 4
                    def temp_func_1(_other = other, _intersection = intersection):
                        if utils.min_degree_difference(_other.dir(), segment.dir()) < MINIMUM_INTERSECTION_DEVIATION:
                            return False
                        _other.split(_intersection, segment, segments)
                        segment.end = _intersection
                        segment.q['severed'] = True
                        return True
                    func[0] = temp_func_1
        if priority <= 3:
            if utils.length(segment.end, other.end) <= ROAD_SNAP_DISTANCE:
                point = other.end
                priority = 3
                def temp_func_2(_other = other, _point = point):
                    segment.end = _point
                    segment.q['severed'] = True
                    links = _other.links['f'] if _other.start_is_backwards() else _other.links['b']
                    are_there_duplicates = False
                    for link in links:
                        terminator_a = utils.equal_vector(link.start, segment.end) and utils.equal_vector(link.end, segment.start)
                        terminator_b = utils.equal_vector(link.start, segment.end) and utils.equal_vector(link.end, segment.start)
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
            distance_to_line = utils.distance_to_line(segment.end, other.start, other.end)
            distance2 = distance_to_line['distance2']
            pointOnLine = distance_to_line['pointOnLine']
            lineProj2 = distance_to_line['lineProj2']
            length2 = distance_to_line['length2']
            if distance2 < ROAD_SNAP_DISTANCE * ROAD_SNAP_DISTANCE and lineProj2 >= 0 and lineProj2 <= length2:
                priority = 2
                def temp_func_3(_other = other, point = pointOnLine):
                    segment.end = point
                    segment.q['severed'] = True
                    if utils.min_degree_difference(_other.dir(), segment.dir()) < MINIMUM_INTERSECTION_DEVIATION:
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
    opposite_direction = Segment.segment_from_existing(root_segment)
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
    