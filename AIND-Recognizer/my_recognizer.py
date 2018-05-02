import warnings
import operator
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    print("test set", test_set)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    words = models.keys()
    for index in range(test_set.num_items):
        probability = dict()

        X, lengths = test_set.get_item_Xlengths(index)

        for word in words:
            try:
                probability[word] = models[word].score(X, lengths)
            except:
                # if the model cannot score the data, then its probability is 0
                probability[word] = float("-inf")
        probabilities.append(probability)

        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        guesses.append(max(probability.items(), key=operator.itemgetter(1))[0])


    return probabilities, guesses
