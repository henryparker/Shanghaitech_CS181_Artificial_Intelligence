# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

def MoveDirection(start_node, end_node):
    from util import Stack
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    if start_node[1] - end_node[1] == -1:
        return n
    if start_node[0] - end_node[0] == -1:
        return e
    if start_node[0] - end_node[0] == 1:
        return w
    if start_node[1] - end_node[1] == 1:
        return s

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    # from game import Directions
    # s = Directions.SOUTH
    # w = Directions.WEST
    # e = Directions.EAST
    # n = Directions.NORTH
    
    path_to_node = []
    expanded = []
    frontier = Stack() #(node, s/w/e/n ,cost)
    frontier.push((problem.getStartState(),[],0))

    while ~frontier.isEmpty():
        (node, path_to_node, cost) = frontier.pop()
        if problem.isGoalState(node):
            return path_to_node
        if node not in expanded:
            expanded = expanded + [node]
            for children in problem.getSuccessors(node):
                (child, child_direction, child_cost) = children
                frontier.push((child, path_to_node + [child_direction], cost + child_cost))
    return failed
    util.raiseNotDefined()

    # path_to_node = Stack()
    # expanded = set()
    # frontier = Stack()
    # frontier.push(problem.getStartState())
    # node_level = list()

    # while ~frontier.isEmpty():
    #     node = frontier.pop()
    #     print(node)
    #     if problem.isGoalState(node):
    #         # for i in path_to_node.list:
    #         #     print(i)
    #         return path_to_node.list
    #     if node == problem.getStartState():
    #         node_level.append((problem.getStartState(), 0, p))
    #         level = 0
    #     else:
    #         for i in node_level:
    #             if i[0] == node:
    #                 level = i[1]
    #                 path_to_node.push(i[2])
    #     expanded.add(node)
    #     origin_length = len(frontier.list)

    #     for child in problem.getSuccessors(node):
    #         if (child[0] not in expanded) and child[0] not in frontier:
    #             frontier.push(child[0])
    #             node_level.append((child[0], level + 1, MoveDirection(node, child[0])))
    #     current_length = len(frontier.list)
    #     if origin_length == current_length:
    #         pre_node = frontier.list[-1]
    #         for i in node_level:
    #             if i[0] == pre_node:
    #                 pre_level = i[1]
    #         # print(level, pre_level)
    #         # if level == 13:
    #         #     for i in path_to_node.list:
    #         #         print(i)
    #         for i in range(level - pre_level + 1):
    #             path_to_node.pop()
    # return failed
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    path_to_node = []
    expanded = []
    frontier = Queue() #(node, s/w/e/n ,cost)
    frontier.push((problem.getStartState(),[],0))

    while ~frontier.isEmpty():
        (node, path_to_node, cost) = frontier.pop()
        if problem.isGoalState(node):
            return path_to_node
        if node not in expanded:
            expanded = expanded + [node]
            for children in problem.getSuccessors(node):
                (child, child_direction, child_cost) = children
                frontier.push((child, path_to_node + [child_direction], cost + child_cost))
    return failed
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    path_to_node = []
    expanded = []
    frontier = PriorityQueue() #(node, s/w/e/n ,cost)
    frontier.push((problem.getStartState(),[],0), 0)

    while ~frontier.isEmpty():
        (node, path_to_node, cost) = frontier.pop()
        if problem.isGoalState(node):
            return path_to_node
        if node not in expanded:
            expanded = expanded + [node]
            for children in problem.getSuccessors(node):
                (child, child_direction, child_cost) = children
                frontier.push((child, path_to_node + [child_direction], cost + child_cost), cost + child_cost)
    return failed
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    path_to_node = []
    expanded = []
    frontier = PriorityQueue() #(node, s/w/e/n ,cost(distance and heuristic))
    frontier.push((problem.getStartState(),[],0), heuristic(problem.getStartState(), problem))

    while ~frontier.isEmpty():
        (node, path_to_node, cost) = frontier.pop()
        if problem.isGoalState(node):
            return path_to_node
        if node not in expanded:
            expanded = expanded + [node]
            for children in problem.getSuccessors(node):
                (child, child_direction, child_cost) = children
                frontier.update((child, path_to_node + [child_direction], cost + child_cost), cost + child_cost + heuristic(child, problem))
    return failed
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
