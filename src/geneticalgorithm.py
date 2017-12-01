import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np
import os
from keras.datasets import cifar10
from keras.datasets import cifar100
import scipy as sp

###########################################################################
####             LOAD PRE-TRAINED MODEL FROM A JSON FILE               ####
###########################################################################
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_names = unpickle("/../cifar-10-batches-py/batches.meta")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.hdf5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')

###########################################################################
####          LOAD THE CIFAR-10 AND CIFAR-100 DATASETS                 ####
###########################################################################

(x_10, y_train), (x_10_test, y_test) = cifar10.load_data()
(x_100, y_train), (x_100_test, y_test) = cifar100.load_data()


###########################################################################
####          HELPER FUNCTIONS TO GENERATE SEED IMAGES                 ####
###########################################################################

'''
    Return a (prediction, score) tuple, representing the
    classification of teh image at the input INDEX in the
    dataset:
     prediciton - a string that represents the class that the image has been
                    classified as.
     score - A normalized certainty value in the range [0, 1.0] representing the
                the certainty of this classification.

'''
def category(index):
    data = np.array([x_10[index]])
    predictions = loaded_model.predict(data)

    pred = label_names[b'label_names'][np.argmax(predictions)]
    return pred, max(predictions[0])

def category_by_image(image):
    data = np.array(image)
    predictions = loaded_model.predict(data)

    pred = label_names[b'label_names'][np.argmax(predictions)]
    return pred, max(predictions[0])


def find_seed(category):
    index = label_names[b'label_names'].index(category)
    max_auto_index = 0
    max_auto_value = 0

    for i in range(10000):
        data = np.array([x_10[i]])
        predictions = loaded_model.predict(data)

        if predictions[0][index] > max_auto_value:
            max_auto_index = i
            max_auto_value = predictions[0][index]
    return max_auto_index, max_auto_value

'''Generates a seed image from the class horse. Chooses the image with
   the highest score'''
index_final, value = find_seed(b'horse')
sp.misc.imsave("thing" + ".png", x_10[index_final])
print(category(index_final))
target_index = 7

class Solution:
    def __init__(self, image):
        self.image = image
        self.score = loaded_model.predict(np.array([image]))[0][target_index]


def choose_parent(population):
    total = np.sum([sol.score for sol in population])
    seed = np.random.uniform(0.0, total)
    sum = 0
    for s in population:
        sum += s.score
        if sum > seed:
            return s

def swap(m, n, k, a1, a2):
    a1[m][n][k], a2[m][n][k] = a2[m][n][k], a1[m][n][k]


'''DO with probability pc'''
def make_offspring(parent1, parent2):
    child1 = parent1.image.copy()
    child2 = parent2.image.copy()
    for _ in range(500):
        m = np.random.random_integers(0, 31)
        n = np.random.random_integers(0, 31)
        k = np.random.random_integers(0, 2)
        swap(m, n, k, child1, child2)
    return (Solution(child1), Solution(child2))


def mutate(sol):
    for _ in range(300):
        m = np.random.random_integers(0, 31)
        n = np.random.random_integers(0, 31)
        k = np.random.random_integers(0, 2)
        r = np.random.random_integers(0, 255)

        sol.image[m][n][k] = r
    sol.score = loaded_model.predict(np.array([sol.image]))[0][target_index]

def genetic(population_size):
    population_tensors = [np.random.random_integers(0, 255, (32, 32, 3)) for _ in range(population_size - 3)]
    population = [Solution(t) for t in population_tensors] + [Solution(x_10[index_final])]
    population.sort(key=lambda s: s.score)

    generations_since_change = 0
    generation = 0

    while generations_since_change < 700:
        max_fitness = max([s.score for s in population])
        print("Max Fitness : " + str(max_fitness) + " | Generation : " + str(generation))

        p1 = choose_parent(population)
        p2 = p1
        while(p1 == p2):
            p2 = choose_parent(population)

        p3 = choose_parent(population)
        p4 = p3
        while(p3 == p4):
            p4 = choose_parent(population)

        child1, child2 = make_offspring(p1, p2)
        child3, child4 = make_offspring(p3, p4)

        for c in [child1, child2, child3, child4]:
            sample = np.random.uniform()
            if sample < 0.15:
                mutate(c)

        population[0], population[1], population[2], population[3] = child1, child2, child3, child4

        if max([child1.score, child2.score]) <= max_fitness:
            generations_since_change += 1
        else:
            generations_since_change = 0

        population.sort(key=lambda s: s.score)
        generation += 1

    population.sort(key=lambda s: s.score, reverse=True)
    print()
    print("Seed Class : " + str(category(index_final)[0]))
    print("Seed Class Index : " + str(label_names[b'label_names'].index(category(index_final)[0])))
    return population[0].image

auto = genetic(500)
sp.misc.imsave("adversarial_demo" + ".png", auto)
print("Resulting Image Classification : " +  str(category_by_image(np.array([auto]))))
