import argparse
from . import run_all

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", dest="workdir", help="The working directory.")
args = parser.parse_args()
workdir = args.workdir
run_all.run(workdir)