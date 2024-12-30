import numpy as np
from collections import deque
from typing import List


def calc_bounds(area):
    """
    Calculates the bounds of area specified by the curve y=S/x
    """
    bounds = np.ones(area, dtype=np.int64) * area
    bounds //= np.arange(1, area+1, dtype=np.int64)

    return bounds


def fill_forward(filled, bounds):
    """
    Used to calculate covered area in forward pass off FillPathing algorithm. More context is given in FillPathing class.
    """
    f_size = len(filled)
    b_size = len(bounds)
    if f_size >= b_size:
        raise ValueError("Bounds must be larger than filled.")

    filled = np.pad(filled, (0, b_size - f_size), mode='constant', constant_values=0)


    filled[0:b_size - 1:2] = np.maximum(filled[0:b_size - 1:2] + 1, bounds[0:b_size - 1:2])

    filled[1:b_size - 1:2] = filled[0:b_size - 1:2]

    if b_size % 2 == 1:
        filled[-1] = 1

    return filled


def fill_backward(filled, bounds):
    """
    Used to calculate covered area in backward pass off FillPathing algorithm. More context is given in FillPathing class.
    """
    f_size = len(filled)
    b_size = len(bounds)
    if f_size >= b_size:
        raise ValueError("Bounds must be larger than filled.")

    filled = np.pad(filled, (0, b_size - f_size), mode='constant', constant_values=1)

    filled[-1:1:-2] = np.maximum(filled[-2:0:-2] + 1, bounds[-2:0:-2])
    filled[-2:0:-2] = filled[-1:1:-2]

    filled[0] = bounds[0]

    return filled


def find_max_x(f, C, lower_bound=1, expansion_factor=2):
    """
    Finds the maximum integer x such that f(x) <= C without a known upper bound.

    :param f: The function to evaluate.
    :param C: The maximum value for f(x).
    :param lower_bound: The lower bound for x.
    :param expansion_factor: Factor by which to expand the range dynamically.
    :return: The maximum integer x such that f(x) <= C.
    """
    # Step 1: Find an upper bound
    upper_bound = lower_bound * expansion_factor
    while f(upper_bound) <= C:
        lower_bound = upper_bound
        upper_bound *= expansion_factor  # Expand the range

    # Step 2: Perform binary search within the bounds
    while lower_bound < upper_bound:
        mid = (lower_bound + upper_bound + 1) // 2  # Use integer division
        if f(mid) <= C:
            lower_bound = mid
        else:
            upper_bound = mid - 1

    return lower_bound


def find_max_odd_x(f, C, lower_bound=1, expansion_factor=2):
    """
    Finds the maximum odd integer x such that f(x) <= C without a known upper bound.

    :param f: The function to evaluate.
    :param C: The maximum value for f(x).
    :param lower_bound: The lower bound for x.
    :param expansion_factor: Factor by which to expand the range dynamically.
    :return: The maximum odd integer x such that f(x) <= C.
    """

    # Step 1: Find an upper bound
    upper_bound = lower_bound * expansion_factor
    lower_bound += (1 - lower_bound % 2)
    upper_bound += (1 - upper_bound % 2)

    while f(upper_bound) <= C:
        lower_bound = upper_bound
        upper_bound *= expansion_factor
        upper_bound += (1 - upper_bound % 2)  # Ensure upper_bound stays odd

    # Step 2: Perform binary search within the bounds, skipping even numbers
    while lower_bound < upper_bound:
        mid = (lower_bound + upper_bound + 1) // 2
        if mid % 2 == 0:
            mid += 1  # Ensure mid is odd

        if f(mid) <= C:
            lower_bound = mid
        else:
            upper_bound = mid - 2  # Adjust by 2 to stay odd

    return lower_bound


def calc_area_forward(filled, M=35):
    """
    Used to find optimal increment for forward pass off FillPathing algorithm. More context is given in FillPathing class.
    """

    p = len(filled) + 1
    priority = p * M

    func = lambda x: np.sum(fill_forward(filled, calc_bounds(x))[0:p:2]) * 2

    new_area = find_max_odd_x(func, priority, p)
    return new_area


def calc_area_backward(filled, M=35):
    """
    Used to find optimal increment for backward pass off FillPathing algorithm. More context is given in FillPathing class.
    """

    p = len(filled) + 1
    priority = p * M

    func = lambda x: np.sum(fill_backward(filled, calc_bounds(x))[x - 1:0:-2]) * 2 + p

    new_area = find_max_odd_x(func, priority, p)
    return new_area


def calc_fill_split(last_area_high, current_moves, M=35):
    p = last_area_high + 1
    priority = p * M

    def opt_func(area_high):
        area_low = int(area_high / 2.62)

        bounds_high = calc_bounds(area_high)
        bounds_low = calc_bounds(area_low)
        bounds_low = np.hstack((bounds_low, np.zeros(area_high - len(bounds_low), dtype=int)))


        reached_in = np.sum(bounds_high[0:p] - np.maximum(bounds_low[0:p], bounds_high[p])) + current_moves + 2*p

        return reached_in

    new_area_high = find_max_x(opt_func, priority, p)
    new_area_low = int(new_area_high / 2.62)

    return new_area_low, new_area_high


class BasePathing:
    """
    Base class for all pathing algorithms. It implements get_direction method to get the next move in the sequence
    defined by the pathing algorithm. Derived classes should override update_direction method to change the self.direction
    variable.
    """

    def __init__(self):
        self.directions = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        self.direction = [0, 1]
        self.direction_index = 0

        self.right = (1, 0)
        self.left = (-1, 0)
        self.up = (0, 1)
        self.down = (0, -1)
        self.moves = 0
        self.absolute_position = [0, 0]

    def get_direction(self):
        self.update_direction()
        if not self.direction:
            return None
        self.moves += 1
        self.absolute_position[0] += self.direction[0]
        self.absolute_position[1] += self.direction[1]
        return self.direction

    def update_direction(self):
        self.direction = self.directions[0]


class FillPathing(BasePathing):
    """
    Pathing that incrementally fills area under the curve y=S/x with bounds x=S and y=S. Complexity is O(S*logS).
    For continuous case filled area is equal to S*(1+log(S)). Algorithm fills the area in forward and backward passes,
    going upward row by row and returning down row by row respectively. It uses functions calc_area_forward and
    calc_area_backward to find the maximum increment of S it can use for this specific movement while ensuring each cell
    is visited within the maximum moves constraint.
    """


    def __init__(self, M=35, max_area=None):
        super().__init__()
        self.M = M

        self.area = 1
        self.max_area = max_area
        self.max_area_reached = False

        self.filled = np.ones(1, dtype=np.int64)

        self.bounds = np.ones(1, dtype=np.int64)

        self.next_directions = deque()

        self.calculate_directions()

    def calculate_directions(self):

        self.area = calc_area_forward(self.filled, self.M)
        if self.max_area and self.area >= self.max_area:
            self.area = self.max_area + (1 - self.max_area % 2)
            self.max_area_reached = True

        self.bounds = calc_bounds(self.area)
        last_filled = self.filled[:]
        self.filled = np.pad(self.filled, (0, self.area - len(self.filled)), mode='constant', constant_values=0)

        for i in range(0, len(self.bounds)-1, 2):
            self.next_directions.extend([self.right] * (self.bounds[i] - self.filled[i]))
            self.next_directions.append(self.up)

            self.filled[i] = np.maximum(self.bounds[i], self.filled[i])

            self.next_directions.extend([self.left] * (self.filled[i] - self.filled[i+1] - 1))
            self.next_directions.append(self.up)
            self.filled[i + 2] += 1

        if self.max_area_reached:
            return

        self.filled = fill_forward(last_filled, self.bounds)

        self.area = calc_area_backward(self.filled, self.M)
        if self.max_area and self.area >= self.max_area:
            self.area = self.max_area + (1 - self.max_area % 2)
            self.max_area_reached = True

        self.bounds = calc_bounds(self.area)
        last_filled = self.filled[:]

        self.next_directions.extend([self.up] * (self.area - len(self.filled)))

        self.filled = np.pad(self.filled, (0, self.area - len(self.filled)), mode='constant', constant_values=1)

        self.next_directions.append(self.right)
        for i in range(len(self.filled)-1, 0, -2):
            self.next_directions.extend([self.right] * (np.maximum(self.filled[i-1] + 1, self.bounds[i-1]) - self.filled[i] - 1))
            self.next_directions.append(self.down)

            self.next_directions.extend([self.left] * (self.bounds[i-1] - self.filled[i-1] - 1))
            self.next_directions.append(self.down)

        self.next_directions.extend([self.right] * (self.bounds[0] - last_filled[0] - 1))
        self.filled = fill_backward(last_filled, self.bounds)

    def update_direction(self):
        if not self.next_directions:
            if self.max_area_reached:
                self.direction = None
                return
            self.calculate_directions()
        self.direction = self.next_directions.popleft()


class FillPathingOptimized(BasePathing):
    """
    Employs a similar approach to FillPathing but skips parts of the world space that is guaranteed to be filled by later
    passes. It fills area between two curves y=S_1/x and y=S_2/x where S_2=S_1*(sqrt(5)-1)/2. Constant (sqrt(5)-1)/2
    comes up due to tiling effect. calc_fill_split calculates next S_1 and S_2 as to ensure maximum moves constraint.
    Unfortunately it is not really accurate. Instead, area_highs might be used to specify a list of S_1 values determined
    experimentally. Each increment starts where last one ends since it's not paramount for it to return back to starting
    position due to tiling.
    """
    def __init__(self, M=35, max_area=None, area_highs: List[int] = None):
        super().__init__()
        self.M = M

        self.area_low = 0
        self.area_high = 1
        self.area_highs = area_highs
        self.max_area = max_area
        self.max_area_reached = False

        self.filled = np.ones(1, dtype=np.int64)

        self.bounds = np.ones(1, dtype=np.int64)

        self.next_iter = False

        self.next_directions = deque()

        self.calculate_directions()

    def calculate_directions(self):

        new_area_low, new_area_high = calc_fill_split(self.area_high, self.moves, self.M)

        if self.area_highs:
            new_area_high = self.area_highs.pop(0)
            new_area_low = int(new_area_high / 2.62)

        if self.max_area and new_area_high >= self.max_area:
            new_area_high = self.max_area
            new_area_low = int(new_area_high/2.62)
            self.max_area_reached = True

        self.area_low, self.area_high = max(self.area_high, new_area_low), new_area_high


        bounds_high = calc_bounds(self.area_high)
        bounds_low = calc_bounds(self.area_low)
        bounds_low = np.hstack((bounds_low, np.zeros(self.area_high - len(bounds_low), dtype=int)))

        root = int(self.area_high ** 0.5)

        self.next_directions.append(self.up)

        if (self.area_high - root) % 2 == 0:
            self.next_directions.append(self.up)

        for i in range(self.area_high - 2 - (1 - (self.area_high - root) % 2), root - 1, -2):
            self.next_directions.extend([self.right] * (bounds_high[i - 1] - bounds_low[i+1] - 1))
            self.next_directions.append(self.up)

            self.next_directions.extend([self.left] * (bounds_high[i - 1] - bounds_low[i-1] - 1))
            self.next_directions.append(self.up)

        num_left = 0
        for i in range(root-1, bounds_low[root], -2):
            self.next_directions.extend([self.right] * (root - bounds_low[i+1] - 1))
            self.next_directions.append(self.up)

            num_left = np.maximum(0, root - bounds_low[i-1] - 1)
            self.next_directions.extend([self.left] * num_left)

            self.next_directions.append(self.up)

        if (root - bounds_low[root]) % 2 == 0:
            self.next_directions.append(self.down)

        self.next_directions.extend([self.right]*(num_left + 1))

        for i in range(root, self.area_high-2, 2):
            self.next_directions.extend([self.down] * (bounds_high[i] - bounds_low[i] - 1))
            self.next_directions.append(self.right)

            self.next_directions.extend([self.up] * (bounds_high[i] - bounds_low[i + 2] - 1))
            self.next_directions.append(self.right)

        if (self.area_high - root) % 2 == 0:
            self.next_directions.append(self.right)

    def update_direction(self):
        if not self.next_directions:
            if self.max_area_reached:
                self.direction = None
                return
            self.calculate_directions()
            self.next_iter = True
        else:
            self.next_iter = False
        self.direction = self.next_directions.popleft()


class FillPathing4(BasePathing):
    def __init__(self, M=35, max_area=None, area_highs: List[int] = None):
        super().__init__()
        self.M = M

        self.area_low = 1
        self.area_high = 9
        self.area_highs = area_highs
        self.max_area = max_area
        self.max_area_reached = False

        self.filled = np.ones(1, dtype=np.int64)

        self.bounds = np.ones(1, dtype=np.int64)

        self.next_iter = False

        self.next_directions = deque()

        self.calculate_directions()


    def calculate_directions(self):


        print(self.area_low, self.area_high)

        bounds_high = calc_bounds(self.area_high)
        bounds_low = calc_bounds(self.area_low)
        bounds_low = np.hstack((bounds_low, np.zeros(self.area_high - len(bounds_low), dtype=int)))

        root = int(self.area_high ** 0.5)

        self.next_directions.append(self.up)

        if (self.area_high - root) % 2 == 0:
            self.next_directions.append(self.up)

        for i in range(self.area_high - 2 - (1 - (self.area_high - root) % 2), root - 1, -2):
            self.next_directions.extend([self.right] * (bounds_high[i - 1] - bounds_low[i+1] - 1))
            self.next_directions.append(self.up)

            self.next_directions.extend([self.left] * (bounds_high[i - 1] - bounds_low[i-1] - 1))
            self.next_directions.append(self.up)

        num_left = 0
        for i in range(root-1, bounds_low[root], -2):
            self.next_directions.extend([self.right] * (root - bounds_low[i+1] - 1))
            self.next_directions.append(self.up)

            num_left = np.maximum(0, root - bounds_low[i-1] - 1)
            self.next_directions.extend([self.left] * num_left)

            self.next_directions.append(self.up)

        if (root - bounds_low[root]) % 2 == 0:
            self.next_directions.append(self.down)

        self.next_directions.extend([self.right]*(num_left + 1))

        for i in range(root, self.area_high-2, 2):
            self.next_directions.extend([self.down] * (bounds_high[i] - bounds_low[i] - 1))
            self.next_directions.append(self.right)

            self.next_directions.extend([self.up] * (bounds_high[i] - bounds_low[i + 2] - 1))
            self.next_directions.append(self.right)

        if (self.area_high - root) % 2 == 0:
            self.next_directions.append(self.right)

        self.area_low = self.area_high
        self.area_high = 2 * self.area_high + 1

    def update_direction(self):
        if not self.next_directions:
            if self.max_area_reached:
                self.direction = None
                return
            self.calculate_directions()
            self.next_iter = True
        else:
            self.next_iter = False
        self.direction = self.next_directions.popleft()