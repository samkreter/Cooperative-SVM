import utils_svm
import sys

def train(inputfileX, inputfileY):
    utils_svm.trainAndStoreSvms(inputfileX, inputfileY, "test")


if __name__ == '__main__':
    if(len(sys.argv) >= 3):
        train(sys.argv[1], sys.argv[2])
