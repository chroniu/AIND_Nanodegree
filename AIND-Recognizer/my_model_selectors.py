import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_bic   = float("inf")
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                actual_model = self.base_model(n_components)

                # based on https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/17
                # n*n + 2*n*d-1., where n is the number of components and d is the number of features
                p = n_components ** 2 + 2 * n_components * len(actual_model.means_[0]) - 1

                bic_value    = -2 * actual_model.score(self.X, self.lengths) # -2 * log L
                + p * np.log(sum(self.lengths)) # p * log N

                if bic_value < best_bic:
                    best_model = actual_model
                    best_bic = bic_value
            except:
                pass
        return best_model
        # TODO implement model selection based on BIC scores
        #raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_dic   = float("-inf")
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                 # based on https://discussions.udacity.com/t/discriminative-information-criterion-formula-words-vs-parameters/250880/5
                actual_model = self.base_model(n_components)
                m = 0

                other_scores = 0.0
                for word, data in self.hwords.items():
                    if not (word == self.this_word):
                        try:
                            m += 1
                            other_scores += actual_model.score(data[0], data[1])
                        except :
                            # the word could not be scored against the actual model
                            pass

                    dic_value = actual_model.score(self.X, self.lengths) - 1/(m) * other_scores
            except :
                # the word being trainned, could not be trainned with n_components components
                continue
            if dic_value > best_dic:
                best_model = actual_model
                best_dic = dic_value

            #    pass
        return best_model

        #raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_model = None
        best_avg   = float("-inf")

        split_method = KFold(3)
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            likehoods = []
            for cv_train_idx, cv_test_idx in split_method.split(self.X):
                try:
                    train, train_len = combine_sequences(cv_train_idx, self.sequences)

                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train, train_len)

                    test, test_len = combine_sequences(cv_test_idx, self.sequences)
                    likehoods.append(hmm_model.score(self.X[cv_test_idx], self.lengths[cv_test_idx]))
                except:
                    pass
            if(len(likehoods) == 0):
                try:
                    hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    likehoods = hmm_model.score(self.X, self.lengths)
                except: # the model could not have been trainned with n_components, so it will be disregarded
                    likehoods = float('-inf')

            else:
                likehoods = np.array(likehoods).mean()

            if likehoods > best_avg:
                best_model = n_components

        return self.base_model(best_model)
        #raise NotImplementedError
