"""
Script for plotting generator's and critic's losses using numpy arrays from directory.

Usage:
    plot_losses.py (--model-dir=PATH) (--output-dir=PATH)
    plot_losses.py --help

Options:
    -m, --model-dir=PATH           Full path to directory where the losses' arrays are saved.
    -o, --output-dir=PATH          Full path to the directory, where result pictures will be.
    -h, --help                     Show this message
"""

import os
import numpy
import docopt

import matplotlib.pyplot as plt


def _run(opts):
    model_dir = opts['--model-dir']
    if not os.path.exists(model_dir):
        print("Full path to the model's directory is invalid: %s" % model_dir)
        return

    output_dir = opts['--output-dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    critic_loss_list = numpy.load(os.path.join(model_dir, "critic_loss.npy"))
    generator_loss_list = numpy.load(os.path.join(model_dir, "generator_loss.npy"))
    fid_score_list = numpy.load(os.path.join(model_dir, "fid_score.npy"))

    # remove first element from lists
    n = 60000 // (64 * 5)

    c_l_l = []
    for i in range(len(critic_loss_list)):
        if i % n == 0 and i % 30 == 0:
            c_l_l.append(critic_loss_list[i])

    g_l_l = []
    for i in range(len(generator_loss_list)):
        if i % n == 0 and i % 30 == 0:
            g_l_l.append(generator_loss_list[i])

    remove_idx = [0]

    fig = plt.figure(figsize=(15, 15))
    plt.plot(numpy.asarray(numpy.delete(c_l_l, remove_idx)))
    plt.plot([0, 100], [0, 0], c='r')
    plt.title("Critic loss, every 30 epochs (start with epoch 30)")
    plt.xlabel("Epoch num (30, 60, 120, ...)")
    plt.ylabel("Loss func")
    fig.savefig(os.path.join(output_dir, "critic_loss.png"))
    plt.close()

    fig = plt.figure(figsize=(15, 15))
    plt.plot(numpy.asarray(numpy.delete(g_l_l, remove_idx)))
    plt.plot([0, 100], [0, 0], c='r')
    plt.title("Generator loss, every 30 epochs (start with epoch 30)")
    plt.xlabel("Epoch num (30, 60, 120, ...)")
    plt.ylabel("Loss func")
    fig.savefig(os.path.join(output_dir, "generator_loss.png"))
    plt.close()

    fig = plt.figure(figsize=(15, 15))
    plt.plot(numpy.asarray(numpy.delete(fid_score_list, remove_idx)))
    plt.plot([0, 100], [0, 0], c='r')
    plt.scatter([90], [fid_score_list[-1]], c='g')
    plt.annotate("%f" % fid_score_list[-1], (90, fid_score_list[-1]), xytext=(10,10), textcoords='offset points')
    plt.title("FID score graphic, every 30 epochs (start with epoch 30)")
    plt.xlabel("Epoch num (30, 60, 120, ...)")
    plt.ylabel("FID score")
    fig.savefig(os.path.join(output_dir, "fid_score.png"))
    plt.close()


if __name__ == "__main__":
    _run(docopt.docopt(__doc__))
