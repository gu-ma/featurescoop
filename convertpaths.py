from featurescoop.config import *
import glob
import pickle
import argparse

# python convertpaths.py --oldpath "/home/guma/Documents/Data/featurescoop"


def main(args):

  files = glob.glob(DATA_PATH + '/**/' + IMAGES_FN, recursive=True)
  for file in files:
    print(file)
    images = pickle.load(open(file, 'rb'))
    print(images[:2])
    images_relative = [i.replace(args.oldpath, '') for i in images]
    print(images_relative[:2])
    pickle.dump(images_relative, open(file, 'wb'))


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Remove old DATA_PATH from the images path')
  parser.add_argument('--oldpath', type=str, help='oldpath', required=True)
  args = parser.parse_args()
  main(args) 