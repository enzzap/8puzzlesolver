import heapq
import math
import time


########################
# BASIC HELPERS
########################

# width calculation
def width_of(state):
    return int(math.isqrt(len(state)))

# prints the state as a grid
def print_state(state):
    w = width_of(state)
    for i in range(len(state)):
        print(state[i], end=" ")
        if (i + 1) % w == 0:
            print()
    print()

# makes the goal states according to the N puzzle you entered
# I have preset the blank to be in the bottom right of the goal states
def make_goal(width):
    size = width * width
    return tuple(list(range(1, size)) + [0])


########################
# HEURISTIC CALCULATION
########################

# literally sets h as 0
def h_uniform(state, goal):
    return 0

# counts the number of tiles that don't match goal state
def h_misplaced(state, goal):
    count = 0
    for i in range(len(state)):
        if state[i] != 0 and state[i] != goal[i]:
            count += 1
    return count

# I use quotient to find the row number and remainder to find column number, giving me coordinates for each number
# I then add the difference between the coordinates of the given state's number and where its supposed to be on goal state to calculate the distance heuristic
def h_manhattan(state, goal):
    w = width_of(state)

    goal_pos = {}
    for i in range(len(goal)):
        goal_pos[goal[i]] = (i // w, i % w)

    dist = 0
    for i in range(len(state)):
        tile = state[i]
        if tile == 0:
            continue
        r1, c1 = i // w, i % w
        r2, c2 = goal_pos[tile]
        dist += abs(r1 - r2) + abs(c1 - c2)

    return dist


########################
# PROBLEM FUNCTIONS
########################

# Just a puzzle solved check, returns true or false
def GOAL_TEST(state, goal):
    return state == goal

# basically generates all possible next moves
# instead of finding neighbors and checking their moves, I can simply just move the blank space in one of the four directions at a time to check.
def EXPAND(state):
    """Return children states by sliding the blank (0)."""
    w = width_of(state)
    z = state.index(0)  # this returns index of where the blank space is located in the tuple. tuple searches are such a lifesaver
    r, c = z // w, z % w    # gives the coordinate of the blank space (converts tuple index to row and column coordinates)

    children = []   # stores new possible states here

    def swap(nz):
        tuple_list = list(state)    # tuple is converted into a list so it can be manipulated
        tuple_list[z], tuple_list[nz] = tuple_list[nz], tuple_list[z]   # swapping with neighbor
        return tuple(tuple_list)    # converted back to tuple so it can't be modified after

    if r > 0:
        children.append(swap(z - w))   # up
    if r < w - 1:
        children.append(swap(z + w))   # down
    if c > 0:
        children.append(swap(z - 1))   # left
    if c < w - 1:
        children.append(swap(z + 1))   # right

    return children


########################
# QUEUEING FUNCTION
########################

def QUEUEING_FUNCTION(nodes, child, g2, goal, heuristic, tie):
    h = heuristic(child, goal)
    f = g2 + h
    heapq.heappush(nodes, (f, tie, child))


########################
# TRACEBACK/PATH
########################

# I'm not sure if this was required for the submission but for debugging I really needed this so I'm leaving it
# Creates a list of states that led to the solution and then reverses it to be printed later and display the solution path
def RECONSTRUCT_SOLUTION(goal_state, parent):
    path = []
    cur = goal_state
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


########################
# GENERAL SEARCH
########################

def general_search(initial_state, goal_state, heuristic):
    start_time = time.perf_counter()    # this is used to track the time taken to solve by a particular heuristic
    nodes = []  # collection of unexplored nodes
    tie = 0
    heapq.heappush(nodes, (0, tie, initial_state))

    best_g = {initial_state: 0} # the smallest number of moves found so far to reach the current state
    parent = {initial_state: None}  # this is to track the parent node for the path maker function

    max_queue_size = 1
    expanded = 0
    

    # loops until no nodes left
    while True:

        # if nodes are empty then it failed
        if not nodes:
            end = time.perf_counter()
            return {
                "solved": False,
                "depth": -1,
                "expanded": expanded,
                "max_queue_size": max_queue_size,
                "time_ms": (end - start_time) * 1000.0,
                "solution_path": []
            }
        
        # pops the best node from the list of unexplored nodes
        f, _, state = heapq.heappop(nodes)
        g = best_g[state]
        expanded += 1

        # returns node and ends timer if GOAL_TEST is true means it is solved
        # also calls the path maker to make a solution path
        if GOAL_TEST(state, goal_state):
            end = time.perf_counter()
            path = RECONSTRUCT_SOLUTION(state, parent)
            return {
                "solved": True,
                "depth": g,
                "expanded": expanded,
                "max_queue_size": max_queue_size,
                "time_ms": (end - start_time) * 1000.0,
                "solution_path": path
            }

        # If not solved, continue expanding nodes
        for child in EXPAND(state):
            g2 = g + 1

            if child not in best_g or g2 < best_g[child]:   # accepts the child node only if there was a shorter way to reach it and updates
                best_g[child] = g2
                parent[child] = state
                tie += 1
                QUEUEING_FUNCTION(nodes, child, g2, goal_state, heuristic, tie)
        
        if len(nodes) > max_queue_size: # adds to the max queue size to count it for just the statistic at the end
            max_queue_size = len(nodes)

########################
# USER INPUT
########################

# inputs the start state as one line and then builds a tuple from the input
# input checks in place to prevent the program from literally dying
def input_start_state():
    while True:
        line = input("Enter the puzzle as ONE line of numbers seperated by spaces (use 0 for blank):\n(NOTE: you can only enter valid n-puzzles like 8,15,24,etc)\n").strip()
        parts = line.split()

        try:
            nums = [int(x) for x in parts]
        except ValueError:
            print("Please enter only integers separated by spaces.\n")
            continue

        size = len(nums)
        w = int(math.isqrt(size))
        if w * w != size:
            print(f"You entered {size} numbers. That is not a perfect square (like 9, 16, 25...). Try again.\n")
            continue

        allowed = set(range(size))
        if set(nums) != allowed or len(nums) != len(set(nums)):
            print(f"Invalid set of numbers. You must use each number 0..{size-1} exactly once.\n")
            continue

        return tuple(nums), w


########################
# MAIN
########################

# hard coded initial states for testing
start = (0, 7, 2,
         4, 6, 1,
         3, 5, 8)

goal = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

# user input state
start, width = input_start_state()
goal = make_goal(width)


# Just displaying all the information to the console
print("Start:")
print_state(start)
print("Goal:")
print_state(goal)

tests = [
    ("Uniform Cost Search", h_uniform),
    ("A* Misplaced", h_misplaced),
    ("A* Manhattan", h_manhattan)
]

results = []

for name, h in tests:
    print("\n==============================")
    print("Method:", name)
    print("==============================")

    result = general_search(start, goal, h)
    results.append((name, result))

    print("Solved:", result["solved"])
    print("Depth:", result["depth"])
    print("Expanded:", result["expanded"])
    print("Max Queue Size:", result["max_queue_size"])
    print("Time (ms):", f'{result["time_ms"]:.1f}')

    if result["solved"]:
        print("\nSolution path:")
        for st in result["solution_path"]:
            print_state(st)

# Shows the method with the fastest solve time and shortest depth
solved_only = [(name, r) for (name, r) in results if r["solved"]]
if solved_only:
    best = min(solved_only, key=lambda x: x[1]["time_ms"])
    print(f"\nFastest method: {best[0]} ({best[1]['time_ms']:.2f} ms, depth {best[1]['depth']})")
