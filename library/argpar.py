#!/usr/bin/env python

import argparse
from argparse import RawTextHelpFormatter


def parse():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('Parameters')
    group.add_argument("-setup", help="Set problem to solve", default="PW3d")
    group.add_argument("-solid_solver_model", help="Set solid solver model", default=None)
    group.add_argument("-v_deg", type=int, default=None)
    group.add_argument("-p_deg", type=int,  default=None)
    group.add_argument("-dt", type=float, default=None)
    group.add_argument("-res_dom", type=float, default=None)
    group.add_argument("-fluid_solver_scheme", type=str, default=None)
    group.add_argument("-incre_v_p_max_it", type=int, default=None)
    group.add_argument("-save_path", type=str, default=None)

    return parser.parse_args()
