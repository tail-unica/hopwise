import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

"""
usage:
    python sentence_transformer.py
        --data_path /home/recsysdatasets/coco_ecir
        --dataset T1

        --item_template "ciao {short_description} ciao2 {long_description}"
        --user_template "ciao {user_id}"

        --augment_user_representation sum
        --augment_item_representation sum

check https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html for optimization on sbert inference
"""


desc = """ For each user and item, this script requests a template to be used from the command line, in which {name} contains the name of the column of the corresponding .user and .item file. It then fills the templates |rows| times, which are inputs for the sentence encoder. The embeddings are then saved with the corresponding IDs of the original dataset in the dataset folder."""  # noqa: E501


def get_sentences(templates, data_path):
    dataset_name = data_path.split("/")[-1]

    try:
        print(f"[+] trying to access {dataset_name}.train_test.user.metadata")
        user_file = os.path.join(data_path, f"{dataset_name}.train_test.user.metadata")
        user_df = pd.read_csv(user_file, sep="\t")
    except ValueError:
        print(f"[+] rollback to {dataset_name}.user")
        user_file = os.path.join(data_path, f"{dataset_name}.user")
        user_df = pd.read_csv(user_file, sep="\t")

    user_df["relative_idx"] = range(len(user_df))
    raw_user2idx = dict(zip(user_df.user_id, user_df.relative_idx))

    # Users
    if templates["user"] is not None:
        user_df["input"] = user_df.apply(lambda row: templates["user"].format(**row), axis=1)
        # add a row with indexes from 0 to n_rows
        user_df = user_df.input.tolist()
    else:
        user_df = None

    # Items
    try:
        print(f"[+] trying to access {dataset_name}.train_valid_test.item.metadata")
        item_file = os.path.join(data_path, f"{dataset_name}.train_valid_test.item.metadata")
        item_df = pd.read_csv(item_file, sep="\t")
    except ValueError:
        print(f"[+] rollback to {dataset_name}.item")
        item_file = os.path.join(data_path, f"{dataset_name}.item")
        item_df = pd.read_csv(item_file, sep="\t")

    item_df["relative_idx"] = range(len(item_df))

    try:
        raw_item2idx = dict(zip(item_df["item_id"], item_df["relative_idx"]))
    except ValueError:
        raw_item2idx = dict(zip(item_df["item_id:token"], item_df["relative_idx"]))

    if templates["item"] is not None:
        item_df["input"] = item_df.apply(lambda row: templates["item"].format(**row), axis=1)
        # add a row with indexes from 0 to n_row
        item_df = item_df.input.tolist()
    else:
        item_df = None

    return user_df, item_df, raw_user2idx, raw_item2idx


def get_embeddings(sentence_encoder, inputs):
    item_encoders_output, user_encoders_output = defaultdict(dict), defaultdict(dict)

    for encoder in sentence_encoder:
        print(f"[+] Encoding with {encoder}")
        model = SentenceTransformer(f"sentence-transformers/{encoder}")

        if inputs["user"] is not None:
            user_embeddings = model.encode(
                inputs["user"], show_progress_bar=True, convert_to_numpy=True, batch_size=4096
            )
            user_encoders_output[encoder] = user_embeddings
        else:
            user_encoders_output[encoder] = list()

        if inputs["item"] is not None:
            item_embeddings = model.encode(inputs["item"], show_progress_bar=True, convert_to_numpy=True)
            item_encoders_output[encoder] = item_embeddings
        else:
            user_encoders_output[encoder] = list()

    return user_encoders_output, item_encoders_output


def save_embeddings(embeddings, data_path, sentence_encoder_name, output_path):
    # save embeddings in numpy format so they can be loaded by hopwise
    for encoder in sentence_encoder_name:
        dataset_name = data_path.split("/")[-1]

        if output_path is not None:
            path = os.path.join(output_path, dataset_name)
            os.makedirs(path, exist_ok=True)

        itememb_file = os.path.join(path, f"{dataset_name}_{encoder}.itememb")
        useremb_file = os.path.join(path, f"{dataset_name}_{encoder}.useremb")

        try:
            user_file = os.path.join(data_path, f"{dataset_name}.train_test.user.metadata")
            raw_users = pd.read_csv(user_file, sep="\t").user_id.tolist()
        except ValueError:
            user_file = os.path.join(data_path, f"{dataset_name}.user")
            raw_users = pd.read_csv(user_file, sep="\t").user_id.tolist()

        item_file = os.path.join(data_path, f"{dataset_name}.train_valid_test.item.metadata")
        raw_items = pd.read_csv(item_file, sep="\t").item_id.tolist()

        itememb_df = pd.DataFrame(
            {"item_embedding_id:token": raw_items, "item_embedding:float_seq": embeddings["item"][encoder].tolist()}
        )
        useremb_df = pd.DataFrame(
            {"user_embedding_id:token": raw_users, "user_embedding:float_seq": embeddings["user"][encoder].tolist()}
        )

        # convert item_embedding:float_seq to string
        itememb_df["item_embedding:float_seq"] = itememb_df["item_embedding:float_seq"].apply(
            lambda x: " ".join(map(str, x))
        )
        useremb_df["user_embedding:float_seq"] = useremb_df["user_embedding:float_seq"].apply(
            lambda x: " ".join(map(str, x))
        )

        # save
        itememb_df.to_csv(itememb_file, sep="\t", index=False)
        print(f"[+] {encoder} itememb saved in {itememb_file}")
        useremb_df.to_csv(useremb_file, sep="\t", index=False)
        print(f"[+] {encoder} itememb saved in {useremb_file}")


def augment_user_embeddings(data_path, embeddings, aggregation, raw_item2idx, raw_user2idx):
    # augment user embeddings with the embeddings of the interacted items on the training set
    dataset_name = data_path.split("/")[-1]
    train_inter = pd.read_csv(os.path.join(data_path, f"{dataset_name}.train.inter"), sep="\t")

    items_users = train_inter.groupby("user_id:token")["item_id:token"].apply(list)

    items_users = items_users.map(lambda items: [raw_item2idx[item] for item in items])
    items_users = items_users.to_dict()
    for encoder in embeddings["user"]:
        if not embeddings["user"][encoder]:
            for user, items in items_users.items():
                embeddings["user"][encoder].append(aggregation(embeddings["item"][encoder][items], axis=0))
        else:
            for user, items in items_users.items():
                embeddings["user"][encoder][raw_user2idx[user]] = embeddings["user"][encoder] + aggregation(
                    embeddings["item"][encoder][items], axis=0
                )

        embeddings["user"][encoder] = np.vstack(embeddings["user"][encoder])

    return embeddings["user"]


def augment_item_embeddings(data_path, embeddings, aggregation, raw_item2idx, raw_user2idx):
    # augment items embeddings with the embeddings of the users that have interacted with that item on the training set
    dataset_name = data_path.split("/")[-1]
    train_inter = pd.read_csv(os.path.join(data_path, f"{dataset_name}.train.inter"), sep="\t")

    # group by items to take the users
    user_items = train_inter.groupby("item_id:token")["user_id:token"].apply(list)
    user_items = user_items.map(lambda users: [raw_user2idx[user] for user in users])
    user_items = user_items.to_dict()

    for encoder in embeddings["item"]:
        if not embeddings["item"][encoder]:
            for item, users in user_items.items():
                embeddings["item"][encoder].append(aggregation(embeddings["user"][encoder][users], dim=0))
        else:
            for user, items in user_items.items():
                embeddings["item"][encoder][raw_item2idx[item]] = embeddings["item"][encoder] + aggregation(
                    embeddings["user"][encoder][items], dim=0
                )

        embeddings["item"][encoder] = np.vstack(embeddings["item"][encoder])

    return embeddings["item"]


aggregators = {
    "sum": np.sum,
    "mean": np.mean,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--data_path", type=str, help="dataset folder")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name where a .item file is present. Specify multiple datasets separated by comma",
    )
    parser.add_argument(
        "--user_template",
        type=str,
        default=None,
        help="example: 'User has the following features {name:token} with {year:float} and {section:token}'",
    )
    parser.add_argument(
        "--item_template",
        type=str,
        default=None,
        help="example: 'Item has the following features {name:token} with {width:float} and {cost:float}'",
    )
    parser.add_argument(
        "--sentence_encoder",
        type=str,
        default="sentence-transformers/sentence-t5-base",
        help="sentence encoder/s to use from theSentenceTransformer Library",
    )
    parser.add_argument(
        "--augment_user_representation",
        type=str,
        default=None,
        help="whether to sum to user embeddings an aggregation of the embeddings of the items the user has interacted with. options: sum, mean",  # noqa: E501
    )
    parser.add_argument(
        "--augment_item_representation",
        type=str,
        default=None,
        help="whether to sum to item embeddings an aggregation of the embeddings of the users the item was interacted by. options: sum, mean",  # noqa: E501
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output path where to save the embeddings. If None, embeddings are saved in the dataset folder",
    )
    args = parser.parse_args()

    # list of sentence encoders to use
    args.sentence_encoder = args.sentence_encoder.split(",")

    for dataset in args.dataset.split(","):
        print(f"[+] Processing dataset {dataset}")

        data_path = os.path.join(args.data_path, dataset)

        templates = {"user": args.user_template, "item": args.item_template}

        user_sentences, item_sentences, raw_user2idx, raw_item2idx = get_sentences(templates, data_path)

        inputs = {"user": user_sentences, "item": item_sentences}
        user_embeddings, item_embeddings = get_embeddings(args.sentence_encoder, inputs)

        embeddings = {"user": user_embeddings, "item": item_embeddings}

        if args.augment_user_representation is not None:
            user_aggregation = aggregators[args.augment_user_representation]
            embeddings["user"] = augment_user_embeddings(
                data_path, embeddings, user_aggregation, raw_item2idx, raw_user2idx
            )

        if args.augment_item_representation is not None:
            item_aggregation = aggregators[args.augment_item_representation]
            embeddings["item"] = augment_item_embeddings(
                data_path, embeddings, item_aggregation, raw_item2idx, raw_user2idx
            )

        save_embeddings(embeddings, data_path, args.sentence_encoder, args.output_path)
