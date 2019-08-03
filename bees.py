"""
Implememntation of the artificial bee colony algorithm (Karaboga, 2005).

An artificial bee colony has the following features:

    workers   -- number of worker bees,
    observers -- number of observer bees,
    solutions -- a set of solutions, one for each worker bee
    alpha     -- a variance factor
    fitness   -- a fitness function
    mutate    -- a mutation function

Each iteration of the algorithm consists of the following steps:

    For each worker:
        Create a mutated copy of the solution (see the Vector.mutate docs).
        If the mutant's fitness is better than that of the solution,
            replace the solution with the mutant
            and refresh its attempt counter.
            If the mutant's fitness is better than the best seen so far,
                replace the current best solution with the mutant.
        If not, decrement the solution's attempt counter.
        
    For each observer:
        Select one of the workers' solutions,
            with probability proportional to its fitness.
        Work on that solution, as in the worker phase.

    For each solution:
        If the solution's attempt counter has hit 0,
            reinitialize the solution as a random vector.
"""

import random
from fitness import func

def log(s):
    # pass
    print (s)

def minimize(f):
    """
    Return a positive number that is higher for lower values of f(x).

    Takes a function f, which must accept one argument and return a number.
    """
    
    def fitness(x):
        fx = f(x)
        if fx >= 0:
            return 1.0 / (fx + 1)
        else:
            return 1.0 + abs(fx)
    return fitness


class Vector:
    
    """
    Solution vector to be optimized by some algorithm.

    Because any functionality that depends on the form of the solution is
    located in this class, swapping the class out allows the ABC algorithm to
    be applied to any solution space. This particular implementation is just a
    toy.

    Instance variables:
    v -- list of floats, each of which is a "feature" of the vector
    """
    
    size = 64
    
    def __init__(self, v):
        """
        Initialize a Vector with some list of values v.
        """
        
        self.v = v
        self.clamp()
        
    @staticmethod
    def new():
        """
        Return a fresh, randomized Vector.
        """

        v = []
        for i in range(Vector.size):
            v.append(random.uniform(0, 10))
        return Vector(v)
    
    @minimize
    def fitness(self):
        """
        Return a positive float representing the desirability of the vector.

        The objective is to minimize the root mean squared deviation.


        length = len(list(self))
        mean = float(sum(self)) / length
        total_squared_deviation = sum(((elt - mean ** 2 for elt in self)))
        return total_squared_deviation / length"""

        num_weights = 64
        equation_inputs = num_weights * [4]
        total_squared_deviation = func(equation_inputs)
        return total_squared_deviation




    def clamp(self):
        """
        Ensure that all values in the vector are within [0, 10] (in-place).
        """
        
        self.v = [max(0, min(10, elt)) for elt in self]

    # Allow indexing and iteration.
    def __getitem__(self, k):
        return self.v[k]
        
    @staticmethod
    def mutate(vectors, n, alpha):
        """
        Construct a mutant Vector v' from the nth vector (v_n) in a population.

        For each index i in the vector v_n:
            Select a random v_m such that m != n.
            Randomly select a phi between -alpha and alpha.
            v'[i] = v_n[i] + phi * (v_n[i] - v_m[i])
        
        In other words, each value in v' is equal to the corresponding value in
        v_n, tweaked either towards or away from the corresponding value in
        some v_m.
        """

        vector = vectors[n]

        mutant_vector = []

        for i, element in enumerate(vector):

            # m is a non-n index in vectors
            m = random.randrange(len(vectors) - 1)
            if m >= n: m += 1

            phi = random.uniform(-alpha, alpha)
            diff = element - vectors[m][i]

            mutant_vector.append(element + phi * diff)
            return Vector(mutant_vector)

    def __str__(self):
        return '[ ' + ', '.join(("%5.15f" % e for e in self)) + ' ]'
    
class Solution:

    """
    Class for keeping track of solution exploration.

    Each Solution has the following attributes:
    vector   -- the actual representation of the solution
    fitness  -- cached fitness value for the 
    attempts -- number of consecutive times the solution can fail to improve
                before being discarded
    """
    
    max_attempts = 10
    
    def __init__(self, vector, fitness, attempts):
        self.vector = vector
        self.fitness = fitness
        self.attempts = attempts

    @staticmethod
    def from_vector(vector):
        """
        Return a Solution for a predetermined solution vector.

        Accept a Vector as a parameter (or an instance of some other class
        that implements the new, mutate, and value methods). Cache its fitness,
        and initialize the maximum number of consecutive failures to improve.
        """

        fitness = vector.fitness()
        attempts = Solution.max_attempts
        return Solution(vector, fitness, attempts)

    @staticmethod
    def new():
        """
        Return a Solution for a new, randomized solution vector.
        """

        vector = Vector.new()
        fitness = vector.fitness()
        attempts = Solution.max_attempts
        return Solution(vector, fitness, attempts)

    def __str__(self):
        return "%8.3f %3d %s" % (self.fitness, self.attempts, str(self.vector))
    
class Hive:

    """
    Represent a colony of artificial bees that explore a problem space.
    """
    
    def __init__(self, workers=10, observers=10, alpha=0.9999999):
        self.solutions = []
        for \
                i in range(workers):
            self.solutions.append(Solution.new())
        self.best = max(self.solutions, key=lambda s: s.fitness)
        self.alpha = alpha
        self.observers = observers
        
    def mutate(self, n):
        vectors = [s.vector for s in self.solutions]
        mutant_vector = Vector.mutate(vectors, n, self.alpha)
        return Solution.from_vector(mutant_vector)
        
    def work_on(self, n):
        current = self.solutions[n]
        mutant = self.mutate(n)
        if mutant.fitness > current.fitness:
            self.solutions[n] = mutant
            log_mark = '!'
            if mutant.fitness > self.best.fitness:
                self.best = mutant
                log_mark = '^'
        else:
            current.attempts -= 1
            log_mark = ' '
        log("work on %3d %s (%2d)" % (n, log_mark, self.solutions[n].attempts))
            
    def worker_phase(self):
        """
        Work on each solution once.
        """
        
        for n in range(len(self.solutions)):
            self.work_on(n)
            
    def observer_phase(self):
        """
        Work on random solutions, weighted to favor the promising ones.
        """

        fitnesses = [s.fitness for s in self.solutions]
        total_fitness = sum(fitnesses)
        for o in range(self.observers):
            i = 0
            r = random.uniform(0, total_fitness)
            while fitnesses[i] < r:
                r -= fitnesses[i]
                i += 1
            self.work_on(i)
                
    def scout_phase(self):
        """
        Reinitialize any solution that hasn't been fruitful recently.
        """
        
        for i, solution in enumerate(self.solutions):
            if solution.attempts <= 0:
                log("reset %2d" % i)
                self.solutions[i] = Solution.new()
                
    def iteration(self):
        """
        Run through each step of the algorithm once.

        Do not return anything; to get the full results, use the run method.
        """

        self.worker_phase()
        self.observer_phase()
        self.scout_phase()
        log(self)

    def run(self, n):
        """
        Iterate through the steps of the algorithm n times.

        Return a Solution object for the best observed solution. This object
        has a Vector (s.vector) with the solution itself, and a fitness value
        (s.fitness) calculated from the vector.
        """
        
        for i in range(n):
            self.iteration()
        return self.best

    def __str__(self):
        lines = []
        for i, s in enumerate(self.solutions):
            lines.append("%4d  %s" % (i, str(s)))
        lines.append("BEST  " + str(self.best))
        return '\n'.join(lines)

if __name__ == '__main__':
    hive = Hive()
    hive.run(5)
