from math import floor, fmod, sqrt
from random import randint

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