import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)


parser = argparse.ArgumentParser()
'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=10, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--cg_weight', type=float, default=8, help='weight of cycle gan')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.01, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=0.005, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
'''
'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--cg_weight', type=float, default=1, help='weight of cycle gan')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=2e-5, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-6, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
'''

# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=4, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=2, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--cg_weight', type=float, default=1, help='weight of cycle gan')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=1e-3, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-4, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')


show_loss = False
show_topk = False

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)


# cgws = [0.1, 0.5, 1, 2, 4, 8, 16, 32]
# rslrs = [1e-3, 5e-4, 2e-4, 1e-4]
# kgelrs = [2e-4, 1e-4, 5e-5, 2e-5]
# l2ws = [1e-4, 1e-5, 5e-6, 1e-6, 5e-7]
# for cgw in cgws:
#     print(">>>>>>> cg_weight=" + str(cgw))
#     args = parser.parse_args()
#     args.cg_weight = cgw
#     data = load_data(args)
#     train(args, data, show_loss, show_topk)
# for l2w in l2ws:
#     print(">>>>>>> l2_weight=" + str(l2w))
#     args = parser.parse_args()
#     args.l2_weight = l2w
#     data = load_data(args)
#     train(args, data, show_loss, show_topk)
# for rslr, kgelr in zip(rslrs, kgelrs):
#     print(">>>>>>> rslr=" + str(rslr) + ', kgelr=' + str(kgelr))
#     args = parser.parse_args()
#     args.lr_rs = rslr
#     args.lr_kge = kgelr
#     data = load_data(args)
#     train(args, data, show_loss, show_topk)