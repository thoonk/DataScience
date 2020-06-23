from typing import List
import math
from typing import Any
from collections import Counter
from typing import NamedTuple, Optional
from typing import Dict, TypeVar
from collections import defaultdict
from typing import NamedTuple, Union, Any


def entropy(class_probabilities: List[float]) -> float:
    """Given a list of class probabilities, compute the entropy"""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p > 0)  # ignore zero probabilities


# label이 클래스명
def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]


def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))


def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)


class Candidate(NamedTuple):
    lang: str
    tweets: bool
    phd: bool
    did_well: bool
    level: Optional[str] = None
    #  level     lang     tweets  phd  did_well


inputs = [Candidate('Java', False, False, False, 'Senior'),
          Candidate('Java', False, True, False, 'Senior'),
          Candidate('Python', False, False, True, 'Mid'),
          Candidate('Python', False, False, True, 'Junior'),
          Candidate('R', True, False, True, 'Junior'),
          Candidate('R', True, True, False, 'Junior'),
          Candidate('R', True, True, True, 'Mid'),
          Candidate('Python', False, False, False, 'Senior'),
          Candidate('R', True, False, True, 'Senior'),
          Candidate('Python', True, False, True, 'Junior'),
          Candidate('Python', True, True, True, 'Senior'),
          Candidate('Python', False, True, True, 'Mid'),
          Candidate('Java', True, False, True, 'Mid'),
          Candidate('Python', False, True, False, 'Junior')
          ]

T = TypeVar('T')  # generic type for inputs


def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute."""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)  # value of the specified attribute
        partitions[key].append(input)  # add input to the correct partition
    return partitions


# print(partition_by(inputs, 'level'))


def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    """Compute the entropy corresponding to the given partition"""
    # partitions consist of our inputs
    partitions = partition_by(inputs, attribute)

    # but partition_entropy needs just the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)


class Leaf(NamedTuple):
    value: Any


class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None


DecisionTree = Union[Leaf, Split]


# 나무를 이용한 input의 분류
def classify(tree: DecisionTree, input: Any) -> Any:
    """classify the input using the given decision tree"""

    # If this is a leaf node, return its value
    if isinstance(tree, Leaf):
        return tree.value

    # Otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values of are subtrees to consider next
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:  # If no subtree for key,
        return tree.default_value  # return the default value.

    subtree = tree.subtrees[subtree_key]  # Choose the appropriate subtree
    return classify(subtree, input)  # and use it to classify the input.


# 나무 구축
def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    # Count target labels
    label_counts = Counter(getattr(input, target_attribute)
                           for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    # If there's a unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # If no split attributes left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)

    # Otherwise split by the best attribute
    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute"""
        return partition_entropy_by(inputs, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # recursively build the subtrees
    subtrees = {attribute_value: build_tree_id3(subset,
                                                new_attributes,
                                                target_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)


tree = build_tree_id3(inputs, ['lang', 'tweets', 'phd', 'did_well'], 'level')
# print(tree)
print("level(lang='Java', tweets=True, phd=False, did_well=True) = "
      + str(classify(tree, Candidate("Java", True, False, True))))
print("level(lang='Python', tweets=False, phd=False, did_well=False) = "
      + str(classify(tree, Candidate("Python", False, False, False))))
