# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodPos = newFood.asList()
        foodnum = len(foodPos)
        # print(newGhostStates[1])
        # print(newFood.asList())
        # print(newGhostStates[0])

        if foodnum == 0:
            return 10000

        score = 10000
        cloestdistance = 10000
        for i in range(foodnum):
            distance = manhattanDistance(newPos, foodPos[i]) + foodnum * 40
            if distance < cloestdistance:
                cloestdistance = distance
        score -= cloestdistance
        for i in range(len(newGhostStates)):
            if manhattanDistance(newPos,childGameState.getGhostPosition(i+1)) <= 1:
                score = 0
        return score


        "*** YOUR CODE HERE ***"
        # return childGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        agentnum = gameState.getNumAgents()
        ghostnum = agentnum - 1

        def max_value(gameState, depth):
            value = -100000
            if gameState.isLose() or gameState.isWin() or depth + 1 >= self.depth:
                return self.evaluationFunction(gameState)
            for direction in gameState.getLegalActions(0):
                value = max(value, min_value(gameState.getNextState(0, direction), depth + 1, 1))
            return value

        def min_value(gameState, depth, ghostindex):
            value = 100000
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            for direction in gameState.getLegalActions(ghostindex):
                if ghostindex < ghostnum:
                    value = min(value, min_value(gameState.getNextState(ghostindex, direction), depth, ghostindex + 1))
                else:
                    value = min(value, max_value(gameState.getNextState(ghostindex, direction), depth))
            return value

        largestscore = -100000
        bestdirection = ''
        for direction in gameState.getLegalActions(0):
            score = min_value(gameState.getNextState(0, direction), 0, 1)
            if score > largestscore:
                largestscore = score
                bestdirection = direction
        return bestdirection
        util.raiseNotDefined()

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentnum = gameState.getNumAgents()
        ghostnum = agentnum - 1

        def max_value(gameState, depth, alpha, beta):
            value = -100000
            if gameState.isLose() or gameState.isWin() or depth + 1 >= self.depth:
                return self.evaluationFunction(gameState)
            for direction in gameState.getLegalActions(0):
                value = max(value, min_value(gameState.getNextState(0, direction), depth + 1, 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(gameState, depth, ghostindex, alpha, beta):
            value = 100000
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            for direction in gameState.getLegalActions(ghostindex):
                if ghostindex < ghostnum:
                    value = min(value, min_value(gameState.getNextState(ghostindex, direction), depth, ghostindex + 1, alpha, beta))
                else:
                    value = min(value, max_value(gameState.getNextState(ghostindex, direction), depth, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        largestscore = -100000
        bestdirection = ''
        alpha = -100000
        beta = 100000
        for direction in gameState.getLegalActions(0):
            score = min_value(gameState.getNextState(0, direction),0, 1, alpha, beta)
            if score > largestscore:
                largestscore = score
                bestdirection = direction
            if score > beta:
                return bestdirection
            alpha = max(alpha, score)
        return bestdirection
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentnum = gameState.getNumAgents()
        ghostnum = agentnum - 1

        def max_value(gameState, depth):
            value = -100000
            if gameState.isLose() or gameState.isWin() or depth + 1 >= self.depth:
                return self.evaluationFunction(gameState)
            for direction in gameState.getLegalActions(0):
                value = max(value, expect_value(gameState.getNextState(0, direction), depth + 1, 1))
            return value

        def expect_value(gameState, depth, ghostindex):
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            directionnum = len(gameState.getLegalActions(ghostindex))
            if directionnum == 0:
                return 0
            valuesum = 0
            for direction in gameState.getLegalActions(ghostindex):
                if ghostindex < ghostnum:
                    value = expect_value(gameState.getNextState(ghostindex, direction), depth, ghostindex + 1)
                else:
                    value = max_value(gameState.getNextState(ghostindex, direction), depth)
                valuesum = valuesum + value
            return float(valuesum) / float(directionnum)

        
        largestscore = -100000
        bestdirection = ''
        for direction in gameState.getLegalActions(0):
            score = expect_value(gameState.getNextState(0, direction), 0, 1)
            if score > largestscore:
                largestscore = score
                bestdirection = direction
        return bestdirection
        util.raiseNotDefined()
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
        first step: value = c1
                           - manhattanDistance to closest dot 
                           - remaining dot number * c2 (bigger than the longest cost to the current closest dot)
        second step: consider eating the scared ghost, if close enough than value is high

        add the first and second
                
        if unscared ghost is in the same position of pacman, value = 0
    """
    "*** YOUR CODE HERE ***"
    # pacman
    currentpacmanPos = currentGameState.getPacmanPosition()
    # food
    currentfood = currentGameState.getFood()
    foodPos = currentfood.asList()
    foodnum = len(foodPos)
    # ghost
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    ghostPos = [currentGameState.getGhostPosition(i + 1) for i in range(len(currentGhostStates))]
    ghostnum = len(currentGhostStates)
    
    # print(currentpacmanPos, foodPos, foodnum, currentScaredTimes, ghostPos, ghostnum)
    value = 10000
    cloestfooddistance = 10000

    if foodnum == 0:
        return 100000

    for i in range(foodnum):
        distance = manhattanDistance(currentpacmanPos, foodPos[i]) + foodnum * 40
        if distance < cloestfooddistance:
            cloestfooddistance = distance
    value = value - cloestfooddistance

    bestscore = 0
    for i in range(ghostnum):
        if currentScaredTimes[i] == 0 and currentpacmanPos == ghostPos[i]:
            return 0
        if currentScaredTimes[i] > 0:
            score = 1 / manhattanDistance(currentpacmanPos, ghostPos[i]) * 1000
            if score > bestscore:
                bestscore = score
    value = value + bestscore

    return value
    util.raiseNotDefined()

    

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        agentnum = gameState.getNumAgents()
        ghostnum = agentnum - 1

        def max_value(gameState, depth, alpha, beta):
            value = -100000
            if gameState.isLose() or gameState.isWin() or depth + 1 >= self.depth:
                return self.evaluationFunction(gameState)
            for direction in gameState.getLegalActions(0):
                value = max(value, min_value(gameState.getNextState(0, direction), depth + 1, 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(gameState, depth, ghostindex, alpha, beta):
            value = 100000
            if gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            for direction in gameState.getLegalActions(ghostindex):
                if ghostindex < ghostnum:
                    value = min(value, min_value(gameState.getNextState(ghostindex, direction), depth, ghostindex + 1, alpha, beta))
                else:
                    value = min(value, max_value(gameState.getNextState(ghostindex, direction), depth, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        largestscore = -100000
        bestdirection = ''
        alpha = -100000
        beta = 100000
        for direction in gameState.getLegalActions(0):
            score = min_value(gameState.getNextState(0, direction),0, 1, alpha, beta)
            if score > largestscore:
                largestscore = score
                bestdirection = direction
            if score > beta:
                return bestdirection
            alpha = max(alpha, score)
        return bestdirection
        util.raiseNotDefined()