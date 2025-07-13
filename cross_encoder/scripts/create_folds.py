import json
import sys
import argparse
from sklearn.model_selection import KFold
import ir_datasets


def read_queries(dataset_id: str):
    """Load queries from ir_datasets using dataset ID"""
    dataset = ir_datasets.load(dataset_id)
    return [query.query_id for query in dataset.queries_iter()]


def split_queries(queries, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    folds = {}
    for i, (train_index, test_index) in enumerate(kf.split(queries)):
        print(f"Fold {i}:")
        train_queries = [queries[idx] for idx in train_index]
        test_queries = [queries[idx] for idx in test_index]
        folds[i] = {'training': train_queries, 'testing': test_queries}
    return folds


def write_to_file(save, folds):
    with open(save, 'w') as f:
        json.dump(folds, f, indent=4)


def main():
    parser = argparse.ArgumentParser("Split queries into k-folds for k-fold CV.")
    parser.add_argument("--dataset", help='ir_datasets dataset ID (e.g., "msmarco-passage/train/trec-dl-2019").', required=True)
    parser.add_argument("--k", help='Number of folds. Default = 5', type=int, default=5)
    parser.add_argument("--save", help='Path to directory where data will be saved.', required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--random'])

    print('Loading queries from ir_datasets...')
    queries = read_queries(dataset_id=args.dataset)
    print(f'[Done]. Loaded {len(queries)} queries.')

    print('Splitting data into {} folds...'.format(args.k))
    folds = split_queries(queries=queries, k=args.k)
    print('[Done].')

    print('Writing to file...')
    write_to_file(folds=folds, save=args.save)
    print('[Done].')


if __name__ == '__main__':
    main()