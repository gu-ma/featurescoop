
import argparse
import os
import shutil
from glob import glob

def main(options):
  dir1 = options.src
  dir2 = options.dst

  predictions = sorted(glob(dir1+'/**/description.p', recursive=True))
  # print(predictions)
  for p in predictions:
    fp_src = p.replace(dir1, '')[1:]
    dir_dst = os.path.join(dir2, fp_src)
    dir_dst = dir_dst.replace(dir_dst.rsplit('/')[-1],'')
    print(dir_dst)
    if not os.path.isdir(dir_dst):
      os.makedirs(dir_dst)
    shutil.copy(p, dir_dst)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--src', type=str, help='data')
  parser.add_argument('--dst', type=str)
  args = parser.parse_args()
  main(args) 

