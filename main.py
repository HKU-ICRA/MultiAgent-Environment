import sys
sys.path.insert(1, './viewer')
sys.path.insert(1, './wrappers')
sys.path.insert(1, './envs')

import logging
import click
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from env_viewer import EnvViewer
from multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments


logger = logging.getLogger(__name__)

def main():

    core_dir = "./"
    envs_dir = './envs',
    xmls_dir = './assets/xmls',

    examine_env("./envs/base.py", {},
                core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                env_viewer=EnvViewer)

if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
