import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    #
    parser.add_argument('--with_cuda',
                        help='test the cuda version or the c version',
                        default=True, type=str2bool)

    #
    parser.add_argument('--num_cells',
                        help='Number of cells on each dimension',
                        default=4, type=int)

    parser.add_argument('--len_cell',
                        help='Length of the cell on each dimension',
                        default=1.0, type=float)

    # parameters for grid searching
    parser.add_argument('--weight_distance',
                        default=1.0, type=float)
    parser.add_argument('--weight_smoothness',
                        default=1.0, type=float)
    parser.add_argument('--weight_curvature',
                        default=1.0, type=float)

    args = parser.parse_args()

    return args
