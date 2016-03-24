import utils_svm
import parser
import sys

def test(filenameX):
    svms = utils_svm.loadSvmsFromFile("workPlease", 8, 1)
    testX = parser.getNumpyArray(filenameX)

    outputs = []
    for x in testX:
        outputs.append(utils_svm.predictMeanBootstrap(svms, 0, x))

    return outputs


if __name__ == '__main__':
    if(len(sys.argv) >= 2):
        outputs = test(sys.argv[1])
        print(outputs)