import numpy as np

class GridSearch:
    
    def __init__(self, params):
        self.param_list = self.__gridSearch_comb(params)

    def __gridSearchRec(self, params, param_list, nkeys, lens, idx, pos):

        if pos >= nkeys:

            p = {}

            for i, k in enumerate(params.keys()):
                p[k] = params[k][idx[i]]

            param_list.append(p)

            return

        for i in range(lens[pos]):
            idx[pos] = i
            self.__gridSearchRec(params, param_list, nkeys, lens, idx, pos + 1)

    def __gridSearch_comb(self, params):

        idx = np.zeros(len(params), dtype = int)
        param_list = []

        nkeys = len(params)
        lens = [len(params[k]) for k in params.keys()]

        self.__gridSearchRec(params, param_list, nkeys, lens, idx, 0)

        return np.array(param_list)