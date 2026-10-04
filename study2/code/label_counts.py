"""Labelled attack / benign training sequences per partition at p_label=0.1 (model seeds 201, 202), using the
exact mask function of train_v3.py. Output: label_counts.json"""
import pickle, json, random, sys, glob
import numpy as np
def make_label_masks(client_train, p_label, seed):
    rng = random.Random(100_003 * seed + int(round(p_label * 1000)))
    out = []
    for tr in client_train:
        ids = sorted(e["anchor_idx"] for e in tr)
        n = min(len(ids), max(2, int(len(ids) * p_label)))
        out.append(frozenset(rng.sample(ids, n)))
    return out
V3 = "/mnt/user-data/outputs/rerun_v3"
files = {"wustl": f"{V3}/phase2_data/partition_wustl_p{{}}.pkl", "toniot": f"{V3}/phase2_data/partition_toniot_p{{}}.pkl",
         "edgeiiot": f"{V3}/phase2_data/partition_edgeiiot_p{{}}.pkl", "cicapt": f"{V3}/heldout/partition_cicapt_p{{}}.pkl",
         "xiiotid": f"{V3}/heldout2/partition_xiiotid_p{{}}.pkl"}
out = {}
for ds, pat in files.items():
    lab_att, lab_all, clients_with_att, tr_att, val_n, val_att = [], [], [], [], None, None
    for ps in range(101, 111):
        P = pickle.load(open(pat.format(ps), "rb")); tr = P["noniid_train"]; va = P["noniid_val"]
        val_n = sum(len(c) for c in va); val_att = sum(e["label"] for c in va for e in c)
        tr_att.append(sum(e["label"] for c in tr for e in c))
        for sd in (201, 202):
            m = make_label_masks(tr, 0.1, sd)
            la = [sum(e["label"] for e in c if e["anchor_idx"] in m[k]) for k, c in enumerate(tr)]
            lab_att.append(sum(la)); lab_all.append(sum(len(x) for x in m)); clients_with_att.append(sum(a > 0 for a in la))
    out[ds] = dict(train_seqs=sum(len(c) for c in tr), train_attack_mean=float(np.mean(tr_att)),
                   labelled_attack_mean=float(np.mean(lab_att)), labelled_attack_range=[int(min(lab_att)), int(max(lab_att))],
                   labelled_total_mean=float(np.mean(lab_all)), clients_with_labelled_attack_mean=float(np.mean(clients_with_att)),
                   val_seqs=val_n, val_attack=val_att)
    print(ds, out[ds])
json.dump(out, open(f"{V3}/study2_summary/label_counts.json", "w"), indent=1)
