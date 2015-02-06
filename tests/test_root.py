import unittest

import ROOT
import numpy as np


class TestRoot(unittest.TestCase):

    def test_root(self):
        print("Writing a tree")

        f = ROOT.TFile("tree.root", "recreate")
        t = ROOT.TTree("name_of_tree", "tree title")

        # create 1 dimensional float arrays (python's float datatype corresponds to c++ doubles)
        # as fill variables
        n = np.zeros(1, dtype=float)
        u = np.zeros(1, dtype=float)

        # create the branches and assign the fill-variables to them
        t.Branch('normal', n, 'normal/D')
        t.Branch('uniform', u, 'uniform/D')

        # create some random numbers, fill them into the fill varibles and call Fill()
        for i in range(10):
            n[0] = ROOT.gRandom.Gaus()
            u[0] = ROOT.gRandom.Uniform()
            t.Fill()

        # write the tree into the output file and close the file
        f.Write()
        f.Close()

if __name__ == '__main__':
    unittest.main()

