import copy
import math
import pickle
import re
import string
import sys

COMMON_ENGLISH_WORDS = {'the', 'be', 'and', 'of', 'to', 'a', 'in', 'have', 'it'}
COMMON_DUTCH_WORDS = {'op', 'wat', 'het', 'en', 'ik', 'je', 'niet', 'dat', 'de'}
TOTAL_STUMPS = 6
MAX_DEPTH = 6


def check_feature_one(words):
    repeated_letters = ['aa', 'ee', 'ii', 'oo', 'uu']
    for word in words:
        word = word.lower()
        for letter in repeated_letters:
            if letter in word:
                return "True"
    return "False"


def check_feature_two(words):
    for word in words:
        word = word.lower()
        if word in COMMON_DUTCH_WORDS:
            return "True"
    return "False"


def check_feature_third(words):
    for word in words:
        word = word.lower()
        if word in COMMON_ENGLISH_WORDS:
            return "True"
    return "False"


def check_feature_fourth(words):
    avg_length = 0
    for word in words:
        avg_length += len(word)
    avg_length = avg_length / len(words)
    if avg_length > 5:
        return "True"
    return "False"


def check_feature_fifth(words):
    for word in words:
        word = word.lower()
        if 'ij' in word:
            return "True"
    return "False"


def check_feature_sixth(words):
    dutch_compound_vowels = ['ae', 'ei', 'au', 'ai', 'eu', 'ie', 'oe', 'ou', 'ui']
    for word in words:
        word = word.lower()
        for compound_vowel in dutch_compound_vowels:
            if compound_vowel in word:
                return "True"
    return "False"


def check_feature_seventh(words):
    for word in words:
        word = word.lower()
        if word[0] == 's':
            return "True"
    return "False"


def check_feature_eighth(words):
    for word in words:
        if not re.match("^[0-9a-zA-Z_-]*$", word):
            return "True"
    return "False"


def check_feature_ninth(words):
    for word in words:
        word = word.lower()
        if 'y' in word:
            return "True"
    return "False"


def check_feature_tenth(words):
    for word in words:
        word = word.lower()
        if word[0] == 'd':
            return "True"
    return "False"


def form_attributes(words):
    attributes = [check_feature_one(words), check_feature_two(words), check_feature_third(words),
                  check_feature_fourth(words), check_feature_fifth(words), check_feature_sixth(words),
                  check_feature_seventh(words), check_feature_eighth(words), check_feature_ninth(words),
                  check_feature_tenth(words)]
    return attributes


def refine_examples(example):
    tab1 = str.maketrans('', '', string.punctuation)
    tab2 = str.maketrans('', '', '0123456789')
    example = example.translate(tab1)
    example = example.translate(tab2)
    words = example.rstrip('\n').split(" ")
    words = ' '.join(words).split()
    return words


def get_weight(examples, weights):
    total_weight = 0
    for example in examples:
        total_weight += weights[example - 1]
    return total_weight


def majority_value(parent_examples, attribute_map, weights):
    if get_weight(attribute_map['target'].intersection(parent_examples), weights) > (
            get_weight(parent_examples, weights) / 2):
        return 'en'
    else:
        return 'nl'


def split(attribute_map, examples, visited, source_entropy, weights):
    max_entropy = -1
    split_attribute = 0
    for i in attribute_map.keys():
        if i not in visited:
            temp = attribute_map[i].intersection(examples)
            if get_weight(temp, weights) == 0:
                x = 0
            else:
                x = get_weight(temp.intersection(attribute_map['target']), weights) / get_weight(temp,
                                                                                                 weights)
            if get_weight(examples, weights) - get_weight(temp, weights) == 0:
                y = 0
            else:
                y = get_weight((examples - temp).intersection(attribute_map['target']), weights) / (
                    get_weight(examples - temp, weights))
            if x == 0:
                part_a = (1 - x) * math.log2(1 / (1 - x))
            elif x == 1:
                part_a = x * math.log2(1 / x)
            else:
                x_2 = (get_weight(temp, weights) - get_weight(temp.intersection(attribute_map['target']),
                                                              weights)) / get_weight(temp, weights)
                part_a = ((x * math.log2(1 / x)) + (x_2 * math.log2(1 / x_2)))
            if y == 0:
                part_b = (1 - y) * math.log2(1 / (1 - y))
            elif y == 1:
                part_b = y * math.log2(1 / y)
            else:
                y_2 = (get_weight(examples - temp, weights) - get_weight(
                    (examples - temp).intersection(attribute_map['target']), weights)) / (
                          get_weight(examples - temp, weights))
                part_b = ((y * math.log2(1 / y)) + (y_2 * math.log2(1 / y_2)))
            remainder = ((get_weight(temp, weights) / get_weight(examples, weights)) * part_a) + (
                    (1 - (get_weight(temp, weights) / get_weight(examples, weights))) * part_b)
            if source_entropy - remainder > max_entropy:
                split_attribute = i
                max_entropy = source_entropy - remainder
                left_entropy = part_a
                right_entropy = part_b
    return split_attribute, left_entropy, right_entropy


class Tree:
    __slots__ = "subtree", "val", "branch"

    def __init__(self, val):
        self.val = val
        self.branch = {}

    def add_branch(self, subtree, label):
        self.branch[label] = subtree


class Decision:

    def weighted_majority(self, hypothesis, weight, input_param):
        total = 0
        for key in hypothesis.keys():
            if self.predict(hypothesis[key], input_param) == 'en':
                total += 1 * weight[key]
            else:
                total += -1 * weight[key]
        if total > 0:
            return 'en'
        else:
            return 'nl'

    def predict(self, root_node, input_param):
        if len(root_node.branch.keys()) == 0:
            return root_node.val
        else:
            root_node = root_node.branch[str(input_param[root_node.val - 1])]
            return self.predict(root_node, input_param)

    def ada_boost(self, examples, k, attribute_map, source_entropy):
        initial_weight = 1 / (len(examples))
        w = [initial_weight] * len(examples)
        dict_hypothesis = {}
        dict_weight = {}
        visited = set()
        visited.add('target')
        for j in range(0, k):
            tree = self.do_decision_tree(attribute_map, examples, source_entropy, visited, examples, w, 1)
            dict_hypothesis[j] = tree
            error = 0
            correct_predicted_set = set()
            for example in examples:
                if example in attribute_map[tree.val]:
                    predicted_val = tree.branch["True"].val
                else:
                    predicted_val = tree.branch["False"].val
                if example in attribute_map['target']:
                    if predicted_val != 'en':
                        error += w[example - 1]
                    else:
                        correct_predicted_set.add(example)
                else:
                    if predicted_val != 'nl':
                        error += w[example - 1]
                    else:
                        correct_predicted_set.add(example)
            for example in correct_predicted_set:
                w[example - 1] = w[example - 1] * (error / (1 - error))
            w = [float(i) / sum(w) for i in w]
            dict_weight[j] = math.log((1 - error) / error)
        return dict_hypothesis, dict_weight

    def do_decision_tree(self, attribute_map, examples, source_entropy, visited, parent_examples, weights,
                         max_depth=None):
        if max_depth is not None:
            if len(visited) - 1 == max_depth:
                if get_weight(attribute_map['target'].intersection(examples), weights) == get_weight(examples,
                                                                                                     weights):
                    return Tree('en')
                elif get_weight(attribute_map['target'].intersection(examples), weights) == 0:
                    return Tree('nl')
                elif len(attribute_map.keys()) == len(visited):
                    return Tree(majority_value(examples, attribute_map, weights))
                return Tree(majority_value(examples, attribute_map, weights))
        if len(examples) == 0:
            return Tree(majority_value(parent_examples, attribute_map, weights))
        elif get_weight(attribute_map['target'].intersection(examples), weights) == get_weight(examples,
                                                                                               weights):
            return Tree('en')
        elif get_weight(attribute_map['target'].intersection(examples), weights) == 0:
            return Tree('nl')
        elif len(attribute_map.keys()) == len(visited):
            return Tree(majority_value(examples, attribute_map, weights))
        else:
            split_result = split(attribute_map, examples, visited, source_entropy, weights)
            split_attribute = split_result[0]
            left_entropy = split_result[1]
            right_entropy = split_result[2]
            dec_tree = Tree(split_attribute)
            attribute_examples = attribute_map[split_attribute]
            left_visited = copy.deepcopy(visited)
            right_visited = copy.deepcopy(visited)
            left_visited.add(split_attribute)
            right_visited.add(split_attribute)
            subtree1 = self.do_decision_tree(attribute_map, examples.intersection(attribute_examples), left_entropy,
                                             left_visited, examples, weights, max_depth)
            if subtree1 is not None:
                dec_tree.add_branch(subtree1, "True")
            subtree2 = self.do_decision_tree(attribute_map, examples - attribute_examples, right_entropy, right_visited,
                                             examples, weights, max_depth)
            if subtree2 is not None:
                dec_tree.add_branch(subtree2, "False")

            return dec_tree

    def print_rec(self, tree, l):
        if len(tree.branch.keys()) == 0:
            print(l + ' ' + str(tree.val))
        else:
            for label in tree.branch.keys():
                print(l + ' ' + str(tree.val))
                self.print_rec(tree.branch[label], l + '---- ' + label + ' ' + str(tree.val))

    def train_decision(self, train_file, train_type):
        attribute_map = {}
        line_number = 1
        with open(train_file, encoding="utf8") as data:
            each_line = data.readline()
            while each_line:
                refined_line = each_line.split("|")
                target_attribute = refined_line[0]
                words = refine_examples(refined_line[1])
                attributes = form_attributes(words)
                for i in range(1, len(attributes) + 1):
                    if i in attribute_map.keys():
                        if attributes[i - 1] == 'True':
                            attribute_map[i].add(line_number)
                    else:
                        attribute_map[i] = set()
                        if attributes[i - 1] == 'True':
                            attribute_map[i].add(line_number)
                if 'target' in attribute_map.keys():
                    if target_attribute == 'en':
                        attribute_map['target'].add(line_number)
                else:
                    attribute_map['target'] = set()
                    if target_attribute == 'en':
                        attribute_map['target'].add(line_number)
                line_number += 1
                each_line = data.readline()
        lines = set()
        for i in range(1, line_number):
            lines.add(i)
        lines = set()
        for i in range(1, line_number):
            lines.add(i)
        w = [1] * len(lines)
        temp = get_weight(attribute_map['target'], w) / get_weight(lines, w)
        source_entropy = (temp * math.log2(1 / temp)) + ((1 - temp) * math.log2(1 / (1 - temp)))
        visited = set()
        visited.add('target')
        if train_type == 'dt':
            tree = self.do_decision_tree(attribute_map, lines, source_entropy, visited, lines, w, MAX_DEPTH)
            return tree
        elif train_type == 'ada':
            dict_hypothesis, dict_weight = self.ada_boost(lines, TOTAL_STUMPS, attribute_map, source_entropy)
            return dict_hypothesis, dict_weight


def main():
    decision = Decision()
    if sys.argv[1] == 'train':
        with open(sys.argv[3], 'wb') as output:
            pickle.dump(decision.train_decision(sys.argv[2], sys.argv[4]), output)
    elif sys.argv[1] == 'predict':
        attributes = []
        with open(sys.argv[3], encoding="utf8") as data:
            each_line = data.readline()
            while each_line:
                words = refine_examples(each_line)
                attributes.append(form_attributes(words))
                each_line = data.readline()

        with open(sys.argv[2], 'rb') as input_model:
            hypothesis = pickle.load(input_model)
            if type(hypothesis) is Tree:
                for i in range(0, len(attributes)):
                    print(decision.predict(hypothesis, attributes[i]))

            else:
                for i in range(0, len(attributes)):
                    print(decision.weighted_majority(hypothesis[0], hypothesis[1], attributes[i]))


if __name__ == '__main__':
    main()
