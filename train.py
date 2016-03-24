import utils
import sys

def train(inputfileX, inputfileY):
    utils.trainAndStoreSvms(inputfileX, inputfileY, "test")


if __name__ == '__main__':
    if(len(sys.argv) >= 3):
        train(sys.argv[1], sys.argv[0])
