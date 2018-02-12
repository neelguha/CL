'''
Agent Coordination v0

There are three 'agents': agent 0, agent 1, and a controller agent.

The controller starts at a starting point. The controller queries each agent for the loss on surrounding points and steps in the best direction. The controller continues this untill it reaches a weight meeting the threshold for both models.

'''

import tensorflow as tf
import argparse
import sys
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from agent import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_DIMENSIONS = 785*10



class Cost:

    def __init__(self, agents):
        self.agents = agents

    def evaluate(self, weight, use_test = False):
        r = Weight(joined=weight)
        w,b = r.split()
        loss = 0
        for i,agent in enumerate(self.agents):
            local_cost, local_accuracy = agent.get_loss_accuracy(w, b, use_test)
            print "Agent-%d: Loss: %.2E Accuracy: %f" % (i, local_cost, local_accuracy)
            loss += local_cost**2
        print "Total Loss: %.2E" % loss
        return loss

# --- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self, agent):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        self.velocity_i  = [random.random()]*NUM_DIMENSIONS
        self.agent = agent
        self.position_i = agent.get_optimum().joint()

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.7  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognative constant
        c2 = 2  # social constant

        r = self.agent.get_gradient()
        gradients = r.joint()
        for i in range(0, NUM_DIMENSIONS):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            #self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social
            self.velocity_i[i] = gradients[i]


    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, NUM_DIMENSIONS):
            self.position_i[i] = self.position_i[i] - 0.001*self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[1]:
                self.position_i[i] = bounds[1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[0]:
                self.position_i[i] = bounds[0]


class PSO():
    def __init__(self, costFunc, agents, bounds, num_particles, maxiter):
        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for agent in agents:
            swarm.append(Particle(agent))

        # begin optimization loop
        i = 0
        while i < maxiter:
            print "------- ITER %d -------" % i
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                print "Particle %d" % j
                swarm[j].evaluate(costFunc)
                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i += 1


        # print final results
        print 'FINAL:'
        costFunc(pos_best_g, True)


def new_parameter(param, var=0.1):
    return param + np.random.normal(0, var,param.shape)


def main(_):
    a0 = Agent(0)
    #a1 = Agent(1)

    opt = a0.get_optimum()
    hes = a0.get_hessian()
    print hes


    '''c = Cost([a0, a1])
    bounds = (-10, 10)  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    PSO(c.evaluate, [a0, a1], bounds, num_particles=2, maxiter=100)'''

def random_search():
    # Create agents
    a0 = Agent(0)
    a1 = Agent(1)
    w_shape = [784, 10]
    b_shape = [10]
    w = np.zeros(w_shape)
    b = np.zeros(b_shape)
    old_loss = sys.float_info.max
    for i in range(1000):
        # Sample new weights centered around current points
        new_biases = []
        new_weights = []
        losses = []
        for _ in range(30):
            new_biases.append(new_parameter(b))
            new_weights.append(new_parameter(w))
            weight_losses = [a0.get_loss(new_weights[-1], new_biases[-1]),
                             a1.get_loss(new_weights[-1], new_biases[-1])]
            losses.append(weight_losses)
        # Pick the best loss
        optimal_index = np.argmin([sum(l) for l in losses])
        if sum(losses[optimal_index]) > old_loss:
            print "%d. No good next step found."
            continue
        old_loss = sum(losses[optimal_index])
        w = new_weights[optimal_index]
        b = new_biases[optimal_index]
        print "%d. Agent 0 loss: %f (%f) \tAgent 1 loss: %f (%f)" % \
              (i, losses[optimal_index][0], a0.get_accuracy(w, b), losses[optimal_index][1], a1.get_accuracy(w, b))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--agent_id', type=int,
                      default=0,
                      help='Agent ID')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)







