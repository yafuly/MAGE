import csv
import os
import sys
from collections import defaultdict
import random
from datasets import load_dataset

set_names = [
    "cmv",
    "yelp",
    "xsum",
    "tldr",
    "eli5",
    "wp",
    "roct",
    "hswag",
    "squad",
    "sci_gen",
]

oai_list = [
    # openai
    "gpt-3.5-trubo",
    "text-davinci-003",
    "text-davinci-002",
]
llama_list = ["_7B", "_13B", "_30B", "_65B"]
glm_list = [
    "GLM130B",
]
flan_list = [
    # flan_t5,
    "flan_t5_small",
    "flan_t5_base",
    "flan_t5_large",
    "flan_t5_xl",
    "flan_t5_xxl",
]

opt_list = [
    # opt,
    "opt_125m",
    "opt_350m",
    "opt_1.3b",
    "opt_2.7b",
    "opt_6.7b",
    "opt_13b",
    "opt_30b",
    "opt_iml_30b",
    "opt_iml_max_1.3b",
]
bigscience_list = [
    "bloom_7b",
    "t0_3b",
    "t0_11b",
]
eleuther_list = [
    "gpt_j",
    "gpt_neox",
]
model_sets = [
    oai_list,
    llama_list,
    glm_list,
    flan_list,
    opt_list,
    bigscience_list,
    eleuther_list,
]

data_dir = sys.argv[1]
dataset = load_dataset("yaful/DeepfakeTextDetect")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
"""
csv_path = f"{data_dir}/train.csv"
train_results = list(csv.reader(open(csv_path,encoding='utf-8-sig')))[1:]
csv_path = f"{data_dir}/valid.csv"
valid_results = list(csv.reader(open(csv_path,encoding='utf-8-sig')))[1:]
csv_path = f"{data_dir}/test.csv"
test_results = list(csv.reader(open(csv_path,encoding='utf-8-sig')))[1:]
"""
train_results = [
    (row["text"], str(row["label"]), row["src"]) for row in list(dataset["train"])
]
valid_results = [
    (row["text"], str(row["label"]), row["src"]) for row in list(dataset["validation"])
]
test_results = [
    (row["text"], str(row["label"]), row["src"]) for row in list(dataset["test"])
]
merge_dict = {
    "train": (train_results, 800),
    "valid": (valid_results, 100),
    "test": (test_results, 100),
}


test_ood_gpt = dataset["test_ood_gpt"]
test_ood_gpt_para = dataset["test_ood_gpt_para"]
test_ood_gpt.to_csv(os.path.join(data_dir, "test_ood_gpt.csv"))
test_ood_gpt_para.to_csv(os.path.join(data_dir, "test_ood_gpt_para.csv"))


# make domain-specific_model-specific (gpt_j)
def prepare_domain_specific_model_specific():
    tgt_model = "gpt_j"
    testbed_dir = f"{data_dir}/domain_specific_model_specific"
    sub_results = defaultdict(lambda: defaultdict(list))
    print("# preparing domain-specific & model-specific ...")
    for name in set_names:
        print(f"## preparing {name} ...")
        for split in ["train", "valid", "test"]:
            split_results, split_count = merge_dict[split]
            count = 0
            for res in split_results:
                info = res[2]
                res = res[:2]
                if name in info:
                    # human-written
                    if res[1] == "1" and count <= split_count:
                        sub_results[name][split].append(res)
                    # machine-generated
                    if tgt_model in info:
                        assert res[1] == "0"
                        sub_results[name][split].append(res)
                    count += 1

        sub_dir = f"{testbed_dir}/{name}"
        os.makedirs(sub_dir, exist_ok=True)
        for split in ["train", "valid", "test"]:
            print(f"{split} set: {len(sub_results[name][split])}")
            rows = sub_results[name][split]
            row_head = [["text", "label"]]
            rows = row_head + rows
            tmp_path = f"{sub_dir}/{split}.csv"
            with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
                csvw = csv.writer(f)
                csvw.writerows(rows)


# make domain_specific_cross_models
def prepare_domain_specific_cross_models():
    testbed_dir = f"{data_dir}/domain_specific_cross_models"
    sub_results = defaultdict(lambda: defaultdict(list))

    print("# preparing domain_specific_cross_models ...")
    for name in set_names:
        print(f"## preparing {name} ...")
        for split in ["train", "valid", "test"]:
            split_results, split_count = merge_dict[split]
            for res in split_results:
                info = res[2]
                res = res[:2]
                if name in info:
                    # human-written
                    if res[1] == "1":
                        sub_results[name][split].append(res)
                    # machine-generated
                    else:
                        sub_results[name][split].append(res)

        sub_dir = f"{testbed_dir}/{name}"
        os.makedirs(sub_dir, exist_ok=True)
        for split in ["train", "valid", "test"]:
            print(f"{split} set: {len(sub_results[name][split])}")
            rows = sub_results[name][split]
            row_head = [["text", "label"]]
            rows = row_head + rows
            tmp_path = f"{sub_dir}/{split}.csv"
            with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
                csvw = csv.writer(f)
                csvw.writerows(rows)


# make cross_domains_model_specific
def prepare_cross_domains_model_specific():
    print("# preparing cross_domains_model_specific ...")
    for model_patterns in model_sets:
        sub_dir = f"{data_dir}/cross_domains_model_specific/model_{model_patterns[0]}"
        os.makedirs(sub_dir, exist_ok=True)
        # model_pattern = dict.fromkeys(model_pattern)
        _tmp = " ".join(model_patterns)
        print(f"## preparing {_tmp} ...")

        ood_pos_test_samples = []
        out_split_samples = defaultdict(list)
        for split in ["train", "valid", "test"]:
            rows = merge_dict[split][0]
            # print(f"Original {split} set length: {len(rows)}")

            out_rows = []
            for row in rows:
                valid = False
                srcinfo = row[2]
                if row[1] == "1":  # appending all positive samples
                    valid = True
                for pattern in model_patterns:
                    if pattern in srcinfo:
                        valid = True
                        break
                if valid:
                    out_rows.append(row)
                    # out_rows.append(row+[srcinfo[0]])

            out_split_samples[split] = out_rows

        for split in ["train", "valid", "test"]:
            random.seed(1)
            rows = out_split_samples[split]
            pos_rows = [r for r in rows if r[1] == "1"]
            neg_rows = [r for r in rows if r[1] == "0"]
            len_neg = len(neg_rows)
            random.shuffle(pos_rows)
            out_split_samples[split] = pos_rows[:len_neg] + neg_rows

        for split in ["train", "valid", "test"]:
            out_rows = [e[:-1] for e in out_split_samples[split]]
            print(f"{split} set: {len(out_rows)} ...")
            # xxx
            tgt_path = f"{sub_dir}/{split}.csv"
            with open(tgt_path, "w", newline="", encoding="utf-8-sig") as f:
                csvw = csv.writer(f)
                csvw.writerows([["text", "label"]] + out_rows)


# make cross_domains_cross_models
def prepare_cross_domains_cross_models():
    print("# preparing cross_domains_cross_models ...")
    testbed_dir = f"{data_dir}/cross_domains_cross_models"
    os.makedirs(testbed_dir, exist_ok=True)
    for split in ["train", "valid", "test"]:
        csv_path = f"{testbed_dir}/{split}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            rows = [row[:-1] for row in merge_dict[split][0]]
            print(f"{split} set: {len(rows)} ...")
            csvw = csv.writer(f)
            csvw.writerows([["text", "label"]] + rows)


# make unseen_models
def prepare_unseen_models():
    print("# preparing unseen_models ...")
    for model_patterns in model_sets:
        sub_dir = f"{data_dir}/unseen_models/unseen_model_{model_patterns[0]}"
        os.makedirs(sub_dir, exist_ok=True)
        _tmp = " ".join(model_patterns)
        print(f"## preparing ood-models {_tmp} ...")

        ood_pos_test_samples = []
        out_split_samples = defaultdict(list)
        for split in ["train", "valid", "test", "test_ood"]:
            data_name = split if split != "test_ood" else "test"
            rows = merge_dict[data_name][0]

            out_rows = []
            for row in rows:
                valid = False
                srcinfo = row[2]
                for pattern in model_patterns:
                    if split != "test_ood":
                        if pattern in srcinfo:
                            valid = False
                            break
                        valid = True
                    else:
                        if pattern in srcinfo:
                            valid = True
                            break
                if valid:
                    out_rows.append(row)

            out_split_samples[split] = out_rows

        random.seed(1)
        test_rows = out_split_samples["test"]
        test_pos_rows = [r for r in test_rows if r[1] == "1"]
        test_neg_rows = [r for r in test_rows if r[1] == "0"]
        len_aug = len(out_split_samples["test_ood"])
        # print(len_aug)
        random.shuffle(test_pos_rows)
        # out_split_samples['test'] = test_pos_rows[len_aug:] + test_neg_rows
        out_split_samples["test_ood"] = (
            test_pos_rows[:len_aug] + out_split_samples["test_ood"]
        )

        for split in ["train", "valid", "test", "test_ood"]:
            out_rows = [e[:-1] for e in out_split_samples[split]]
            print(f"{split} set: {len(out_rows)}")

            tgt_path = f"{sub_dir}/{split}.csv"
            with open(tgt_path, "w", newline="", encoding="utf-8-sig") as f:
                csvw = csv.writer(f)
                csvw.writerows([["text", "label"]] + out_rows)


# make unseen_domains
def prepare_unseen_domains():
    print("# preparing unseen_domains ...")

    testbed_dir = f"{data_dir}/unseen_domains"
    sub_results = defaultdict(lambda: defaultdict(list))

    for name in set_names:
        sub_dir = f"{data_dir}/unseen_domains/unseen_domain_{name}"
        os.makedirs(sub_dir, exist_ok=True)

        print(f"## preparing ood-domains {name} ...")

        ood_pos_test_samples = []
        out_split_samples = defaultdict(list)
        for split in ["train", "valid", "test", "test_ood"]:
            data_name = split if split != "test_ood" else "test"
            rows = merge_dict[data_name][0]

            out_rows = []
            for row in rows:
                srcinfo = row[2]
                valid = True if name in srcinfo else False
                valid = not valid if split != "test_ood" else valid
                if valid:
                    out_rows.append(row)

            out_split_samples[split] = out_rows

        for split in ["train", "valid", "test", "test_ood"]:
            out_rows = [e[:-1] for e in out_split_samples[split]]
            print(f"{split} set: {len(out_rows)}")
            tgt_path = f"{sub_dir}/{split}.csv"
            with open(tgt_path, "w", newline="", encoding="utf-8-sig") as f:
                csvw = csv.writer(f)
                csvw.writerows([["text", "label"]] + out_rows)


# prepare 6 testbeds
prepare_domain_specific_model_specific()
print("-" * 100)
prepare_domain_specific_cross_models()
print("-" * 100)
prepare_cross_domains_model_specific()
print("-" * 100)
prepare_cross_domains_cross_models()
print("-" * 100)
prepare_unseen_models()
print("-" * 100)
prepare_unseen_domains()
print("-" * 100)
