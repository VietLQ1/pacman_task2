import heapq
import abc
import os
import time

class SearchAlgorithm(abc.ABC):
    @abc.abstractmethod
    def search(self, initial_state):
        pass

    @abc.abstractmethod
    def is_goal_state(self, state):
        pass

    @abc.abstractmethod
    def get_successors(self, state):
        pass

    @abc.abstractmethod
    def is_valid_position(self, x, y):
        pass

    @abc.abstractmethod
    def get_next_state(self, state, next_position, action):
        pass

    @abc.abstractmethod
    def extract_actions(self, final_state):
        pass

class Utils:
    @staticmethod
    def distance(point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    @staticmethod
    def heuristic_func(state):
        pacman_position = state.pacman_position
        food_points = state.food_points
        if not food_points:
            return 0  # If no food points remaining, heuristic value is 0
        return min(Utils.distance(pacman_position, food_point) for food_point in food_points) + len(food_points)

class GameState:
    def __init__(self, pacman_position, food_points, parent_state=None, parent_action=None, cost=0):
        self.pacman_position = pacman_position
        self.food_points = food_points
        self.parent_state = parent_state
        self.parent_action = parent_action
        self.cost = cost  # Initialize cost

    def __lt__(self, other):
        return self.cost < other.cost
    def __gt__(self, other):
        return self.cost > other.cost
    def __eq__(self, other):
        return isinstance(other, GameState) and self.pacman_position == other.pacman_position and self.food_points == other.food_points

    def __hash__(self):
        return hash((self.pacman_position, tuple(sorted(self.food_points))))


class Maze:
    def __init__(self, layout_file):
        self.layout = self.load_layout(layout_file)
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])

    def load_layout(self, layout_file):
        with open(layout_file, 'r') as file:
            layout = [line.strip() for line in file]
        return layout

    def get_initial_state(self):
        pacman_position = None
        food_points = []

        for i in range(self.rows):
            for j in range(self.cols):
                if self.layout[i][j] == 'P':
                    pacman_position = (i, j)
                elif self.layout[i][j] == '.':
                    food_points.append((i, j))

        return GameState(pacman_position, food_points)
    
    def print_maze(self, pacman_position, food_positions):
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == pacman_position:
                    print('P', end='')
                elif (i, j) in food_positions:
                    print('.', end='')
                elif self.layout[i][j] == 'P' or self.layout[i][j] == '.':
                    print(' ', end='')
                else:
                    print(self.layout[i][j], end='')  # Print original maze layout
            print()

class UCS(SearchAlgorithm):  
    def __init__(self, maze):
        self.maze = maze
        self.__name__ = "UCS"

    def search(self, initial_state):
        frontier = [(0, initial_state)]
        explored = set()

        while frontier:
            cost, current_state = heapq.heappop(frontier)
            if self.is_goal_state(current_state):
                return self.extract_actions(current_state), cost

            if hash(current_state) not in explored:
                explored.add(hash(current_state))
                for action, next_state, step_cost in self.get_successors(current_state):
                    new_cost = cost + step_cost
                    heapq.heappush(frontier, (new_cost, next_state))

        return None, float('inf')  # No solution found

    def is_goal_state(self, state):
        # Check if all food points have been collected and all corners visited
        return not state.food_points

    def get_successors(self, state):
        successors = []
        x, y = state.pacman_position

        # Define actions and their effects
        actions = [(0, 1, 'East'), (0, -1, 'West'), (1, 0, 'South'), (-1, 0, 'North'), (0, 0, 'Stop')]
        for dx, dy, action in actions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                next_state = self.get_next_state(state, (new_x, new_y), action)  # Pass action
                step_cost = 1  # Uniform cost for all actions
                successors.append((action, next_state, step_cost))

        return successors

    def is_valid_position(self, x, y):
        return 0 <= x < self.maze.rows and 0 <= y < self.maze.cols and self.maze.layout[x][y] != '%'

    def get_next_state(self, state, next_position, action):
        pacman_position = next_position
        food_points = [point for point in state.food_points if point != next_position]
        cost = state.cost + 1  # Update cost
        return GameState(pacman_position, food_points, state, action, cost)  # Update parent_action with action

    def extract_actions(self, final_state):
        # Trace back actions from final state to initial state
        actions = []
        current_state = final_state
        while current_state.parent_action:
            actions.append(current_state.parent_action)
            current_state = current_state.parent_state
        return actions[::-1]


class A_Star(SearchAlgorithm):
    def __init__(self, maze, heuristic):
        self.maze = maze
        self.heuristic = heuristic
        self.__name__ = "A*"

    def search(self, initial_state):
        frontier = [(self.heuristic(initial_state), 0, initial_state)]
        explored = set()

        while frontier:
            _, cost, current_state = heapq.heappop(frontier)
            if self.is_goal_state(current_state):
                return self.extract_actions(current_state), cost

            if hash(current_state) not in explored:
                explored.add(hash(current_state))
                for action, next_state, step_cost in self.get_successors(current_state):
                    new_cost = cost + step_cost
                    priority = new_cost + self.heuristic(next_state)
                    heapq.heappush(frontier, (priority, new_cost, next_state))

        return None, float('inf')  # No solution found
    
    def is_goal_state(self, state):
        # Check if all food points have been collected and all corners visited
        return not state.food_points

    def get_successors(self, state):
        successors = []
        x, y = state.pacman_position

        # Define actions and their effects
        actions = [(0, 1, 'East'), (0, -1, 'West'), (1, 0, 'South'), (-1, 0, 'North'), (0, 0, 'Stop')]
        for dx, dy, action in actions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                next_state = self.get_next_state(state, (new_x, new_y), action)  # Pass action
                step_cost = 1  # Uniform cost for all actions
                successors.append((action, next_state, step_cost))

        return successors

    def is_valid_position(self, x, y):
        return 0 <= x < self.maze.rows and 0 <= y < self.maze.cols and self.maze.layout[x][y] != '%'

    def get_next_state(self, state, next_position, action):
        pacman_position = next_position
        food_points = [point for point in state.food_points if point != next_position]
        cost = state.cost + 1  # Update cost
        return GameState(pacman_position, food_points, state, action, cost)  # Update parent_action with action

    def extract_actions(self, final_state):
        # Trace back actions from final state to initial state
        actions = []
        current_state = final_state
        while current_state.parent_action:
            actions.append(current_state.parent_action)
            current_state = current_state.parent_state
        return actions[::-1]
    


class PacmanGame:
    @staticmethod
    def console_visualization(search_algorithm, actions):
        if not actions:
            print("No solution.")
            return

        current_state = search_algorithm.maze.get_initial_state()
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
        search_algorithm.maze.print_maze(current_state.pacman_position, current_state.food_points)
        for action in actions:
            # Update current_state based on action
            for dx, dy, action_name in [(0, 1, 'East'), (0, -1, 'West'), (1, 0, 'South'), (-1, 0, 'North'), (0, 0, 'Stop')]:
                if action_name == action:
                    new_x, new_y = current_state.pacman_position[0] + dx, current_state.pacman_position[1] + dy
                    if search_algorithm.is_valid_position(new_x, new_y):
                        current_state = GameState((new_x, new_y), [point for point in current_state.food_points if point != (new_x, new_y)], current_state, action, current_state.cost + 1)
                        #break
            time.sleep(0.05)  # Pause
            # Print the maze on current state
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
            search_algorithm.maze.print_maze(current_state.pacman_position, current_state.food_points)
        print(search_algorithm.__name__ + " Algorithm:")
        print("-> ".join(actions))
        print("Game Completed.")


    @staticmethod
    def execute(layout_file, algorithm):
        maze = Maze(layout_file)
        initial_state = maze.get_initial_state()

        if algorithm == 'UCS':
            search_algorithm = UCS(maze)
        elif algorithm == 'A*':
            heuristic = Utils.heuristic_func
            search_algorithm = A_Star(maze, heuristic)
        else:
            print("Invalid algorithm.")
            return

        actions, total_cost = search_algorithm.search(initial_state)
        PacmanGame.console_visualization(search_algorithm, actions)
        print("Total cost:", total_cost)

file_path = "pacman_layouts\\bigMaze.lay"
algorithm = "A*"  # or "UCS"
PacmanGame.execute(file_path, algorithm)