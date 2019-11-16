import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    train_data = append_kg(args, train_data, kg, 'train')
    eval_data = append_kg(args, eval_data, kg, 'eval')
    test_data = append_kg(args, test_data, kg, 'test')
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, kg

def append_kg(args, rs, kg, mode):
    print('appending kg for ' + mode + ' dataset...')
    npy_path = '../data/' + args.dataset + '/rs_kg_' + mode + '.npy'
    if not os.path.exists(npy_path):
        kg_dic = {}
        for k in kg:
            if k[0] not in kg_dic:
                kg_dic[k[0]] = []
            kg_dic[k[0]].append(k)
        data = []
        for r in rs:
            ks = kg_dic[r[1]]
            for k in ks:
                data.append([r[0], r[1], r[2], k[0], k[1], k[2]])
        data = np.array(data)
        np.save(npy_path, data)
        return data
    else:
        return np.load(npy_path)

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg = np.load(kg_file + '.npy')
    else:
        kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg)

    n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
    n_relation = len(set(kg[:, 1]))

    return n_entity, n_relation, kg
