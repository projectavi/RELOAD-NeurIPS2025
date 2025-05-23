import json
import argparse
import os
import pathlib
import numpy as np
import pandas as pd
from utils import get_targeted_classes
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsvpath', type=str, default='./results.tsv', help='path of tsv file (dont add .tsv)')
    parser.add_argument('--dirpath', type=str, default='/scratch/ssd004/scratch/newatiaa/logs/', help='root directory of saved results')
    parser.add_argument('--out_dir', type=str, default='./results', help='output directory for the results')
    return parser.parse_args()

def parse_pretrain_dirname(dirname, preargnamelist):
    arglist = dirname.split('_')
    
    assert(len(arglist) == len(preargnamelist))
    pretrain_args = {}
    for idx, name in enumerate(preargnamelist):
        pretrain_args[name] = arglist[idx]
    return pretrain_args

def parse_unlearn_dirname(dirname, unargnamelist):
    arglist = dirname.split('_')
    assert(len(arglist) >= 2)
    
    un_args = dict((arg, '') for arg in unargnamelist)
    if arglist[0] == 'Naive':
        un_args['unlearn_method'], un_args['exp_name'] = arglist[0], arglist[1]
        un_args['deletion_size'] = 0
        # if un_args['exp_name'] == 'pretrainmodel':
        return un_args
    un_args['deletion_size'], un_args['unlearn_method'], un_args['exp_name'] = arglist[0], arglist[1], arglist[2]
    # if arglist[0] == 'Naive':
        # print(un_args)
        # exit(0)
    if arglist[1] in ['EU', 'CF', 'Mixed', 'Scrub', 'BadT', 'SSD', 'AscentLearn', 'ScrubNew', 'ALnew']:
        assert(len(arglist) >= 5)
        un_args['train_iters'], un_args['k'] = arglist[3], arglist[4] 
    if arglist[1] in ['Mixed']:
        assert(len(arglist) >= 6)
        un_args['factor'] = arglist[5]
    if arglist[1] in ['AscentLearn', 'ALnew']:
        assert(len(arglist) >= 6)
        un_args['ascLRscale'] = arglist[5]
    if arglist[1] == 'InfRe':
        assert(len(arglist) >= 8)
        un_args['msteps'], un_args['rsteps'], un_args['ascLRscale'] = arglist[5], arglist[6], arglist[7]
    if arglist[1] == 'Scrub':
        assert(len(arglist) >= 8)
        un_args['kd_T'], un_args['alpha'], un_args['msteps'] = arglist[5:8]
    if arglist[1] == 'ScrubNew':
        assert(len(arglist) >= 8)
        un_args['kd_T'], un_args['alpha'], un_args['ascLRscale'] = arglist[5:8]
    if arglist[1] == 'SSD':
        assert(len(arglist) >= 7)
        un_args['SSDdampening'], un_args['selectwt'] = arglist[5], arglist[6]
    if arglist[1] == 'Reload':
        if arglist[4] == "xavier" or arglist[4] == "kaiming":
            un_args['repair_epochs'], un_args['replacement_strategy'], strat_pt_2, un_args['threshold'], un_args['rlr'], un_args['alr'] = arglist[3:9]
            un_args['replacement_strategy'] = un_args['replacement_strategy'] + '_' + strat_pt_2
        else:
            un_args['repair_epochs'], un_args['replacement_strategy'], un_args['threshold'], un_args['rlr'], un_args['alr'] = arglist[3:8]
    return un_args

def compute_accuracy(preds, y):
    return np.equal(np.argmax(preds, axis=1), y).mean()

def parse_unpath(un_path, pre_args, un_args, args, headers):

    ret = dict((key, '') for key in headers)
    ret.update(pre_args)
    ret.update(un_args)

    if os.path.exists(un_path + f'/preds_train.npy'):
        tr_preds = np.load(un_path + f'/preds_train.npy')
    else:
        # print('No train preds for', un_path)
        return None

    if os.path.exists(un_path + f'/targetstrain.npy'):
        tr_y = np.load(un_path + f'/targetstrain.npy')
    else:
        # print('No train targets for', un_path)
        return None

    if os.path.exists(un_path + f'/preds_test.npy'):
        te_preds = np.load(un_path + f'/preds_test.npy')
    else:
        # print('No test preds for', un_path)
        return None

    if os.path.exists(un_path + f'/targetstest.npy'):
        te_y = np.load(un_path + f'/targetstest.npy')
    else:
        # print('No test targets for', un_path)
        return None

    if os.path.exists(un_path + f'/unlearn_time.npy'):
        un_time = np.load(un_path + f'/unlearn_time.npy')
    else:
        # print('No unlearn time for', un_path)
        return None


    ret['unlearn_time'] = un_time
    forget_idx = np.load(args.dirpath+'/'+pre_args['dataset']+'_'+pre_args['dataset_method']+'_'+pre_args['forget_set_size']+'_manip.npy')
    if un_args['deletion_size'] != 0:
        delete_idx = np.load(args.dirpath+'/'+pre_args['dataset']+'_'+pre_args['dataset_method']+'_'+pre_args['forget_set_size']+'_'+un_args['deletion_size']+'_deletion.npy')
    ret['train_clean_acc'] = compute_accuracy(tr_preds, tr_y) 
    delete_acc, delete_err = 0.0, 101.0
    if pre_args['dataset_method'] == 'poisoning':
        try:
            tr_adv_preds = np.load(un_path + f'/preds_adv_train.npy')
            tr_adv_y = np.load(un_path + f'/targetsadv_train.npy')
            tr_wrong = np.zeros(tr_adv_y.shape)
            te_adv_preds = np.load(un_path + f'/preds_adv_test.npy')
            te_adv_y = np.load(un_path + f'/targetsadv_test.npy')
        except:
            return None
        forget_acc = compute_accuracy(tr_adv_preds[forget_idx], tr_adv_y[forget_idx])
        if un_args['deletion_size'] != 0: 
            delete_err = compute_accuracy(tr_adv_preds[delete_idx], tr_wrong[delete_idx])
            delete_acc = compute_accuracy(tr_adv_preds[delete_idx], tr_adv_y[delete_idx])
        test_acc = compute_accuracy(te_adv_preds, te_adv_y)
        forget_clean_acc = compute_accuracy(tr_preds[forget_idx], tr_y[forget_idx])
        test_clean_acc = compute_accuracy(te_preds, te_y)
        # print(forget_acc, test_acc, forget_clean_acc, test_clean_acc)
        ret['delete_acc'], ret['delete_err'], ret['manip_acc'], ret['test_acc'], ret['manip_clean_acc'], ret['test_clean_acc'] =\
         delete_acc, delete_err, forget_acc, test_acc, forget_clean_acc, test_clean_acc


    if pre_args['dataset_method'] == 'labelrandom':
        forget_acc = compute_accuracy(tr_preds[forget_idx], tr_y[forget_idx])
        test_acc = compute_accuracy(te_preds, te_y)
        # print(forget_acc, test_acc)
        ret['forget_acc'], ret['test_acc'] = forget_acc, test_acc

    if pre_args['dataset_method'] == 'interclasslabelswap':
        classes = get_targeted_classes(pre_args['dataset'])
        te_class_idxes = np.concatenate((np.nonzero(te_y == classes[0]), np.nonzero(te_y == classes[1])), axis=1).squeeze()
        retain_idxes =np.setdiff1d(np.arange(len(te_y)), te_class_idxes)
        forget_acc = compute_accuracy(tr_preds[forget_idx], tr_y[forget_idx])
        tr_wrong = tr_y
        tr_wrong[tr_y == classes[0]] = classes[1]
        tr_wrong[tr_y == classes[1]] = classes[0]
        if un_args['deletion_size'] != 0: 
            delete_acc = compute_accuracy(tr_preds[delete_idx], tr_y[delete_idx])
            delete_err = compute_accuracy(tr_preds[delete_idx], tr_wrong[delete_idx])
        test_acc = compute_accuracy(te_preds[te_class_idxes], te_y[te_class_idxes])
        test_retain_acc = compute_accuracy(te_preds[retain_idxes], te_y[retain_idxes])
        # print(forget_acc, test_acc, test_retain_acc) 
        ret['delete_acc'], ret['delete_err'], ret['manip_acc'], ret['test_acc'], ret['test_retain_acc']  = delete_acc, delete_err, forget_acc, test_acc, test_retain_acc

    return ret


if __name__ == '__main__':
    # datasets = ['CIFAR10', 'CIFAR100', 'PCAM', 'Pneumonia', 'DermaNet']
    # model = ['resnet9', 'resnetwide28x10']
    # dataset_method = ['labelrandom', 'labeltargeted', 'poisoning']
    unlearn_method = ['Naive', 'CF', 'BadT', 'Scrub', 'SSD', 'Reload']
    preargnamelist = ['dataset', 'model', 'dataset_method', 'forget_set_size', 'patch_size', 'pretrain_iters', 'pretrain_lr']
    unargnamelist = ['unlearn_method', 'exp_name', 'train_iters', 'k', 'factor', 'kd_T', 'gamma', 'alpha', 'msteps', 'SSDdampening', 'selectwt', 'rsteps', 'ascLRscale', 'alr', 'rlr', 'threshold', 'replacement_strategy']
    metricslist = ['delete_acc', 'delete_err', 'manip_acc', 'test_acc', 'manip_clean_acc', 'test_clean_acc', 'test_retain_acc', 'cost']
    headers = preargnamelist + unargnamelist + metricslist

    unargs_per_method = {
        "Naive": ["unlearn_method", "exp_name"],
        "EU": ["unlearn_method", "exp_name", "train_iters", "k"],
        "CF": ["unlearn_method", "exp_name", "train_iters", "k"],
        "Mixed": ["unlearn_method", "exp_name", "train_iters", "k", "factor"],
        "Scrub": ["unlearn_method", "exp_name", "train_iters", "k", "kd_T", "alpha", "msteps"],
        "BadT": ["unlearn_method", "exp_name", "train_iters", "k"],
        "SSD": ["unlearn_method", "exp_name", "train_iters", "k", "SSDdampening", "selectwt"],
        "AscentLearn": ["unlearn_method", "exp_name", "train_iters", "k", "ascLRscale"],
        "ScrubNew": ["unlearn_method", "exp_name", "train_iters", "k", "kd_T", "alpha", "ascLRscale"],
        "ALnew": ["unlearn_method", "exp_name", "train_iters", "k", "ascLRscale"],
        "InfRe": ["unlearn_method", "exp_name", "train_iters", "k", "msteps", "rsteps", "ascLRscale"],
        "Reload": ["unlearn_method", "exp_name", "repair_epochs", "replacement_strategy", "threshold", "rlr", "alr"]
    }

    method_to_retain_measure = {
        "poisoning": "test_clean_acc",
        "interclasslabelswap": "test_retain_acc"
    }

    # Map methods to markers, do not reuse
    method_to_markers = {
        "Naive": 'o',
        "EU": 's',
        "CF": '^',
        "RewoD": 'd',
        "Scrub": '*',
        "BadT": 'x',
        "SSD": 'v',
        "Reload": '8'
    }

    method_to_colours = {
        "Naive": 'red',
        "EU": 'blue',
        "CF": 'green',
        "RewoD": 'black',
        "Scrub": 'olive',
        "BadT": 'brown',
        "SSD": 'pink',
        "Reload": 'purple'
    }



    args = parse_args()
    rows = []
    skipped = []

    A = 2  # Want figures to be A2
    plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 60, "lines.linewidth": 8, "lines.markersize": 40})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # plt.rcParams.update({'mathtext.default':  'regular' })

    baselines = {}

    # FOR TESTING
    # COUNT = -1

    wrapped_outer = tqdm(next(os.walk(args.dirpath))[1], desc='Pretrain dirs', unit='dir')
    for dirname in wrapped_outer:

        # COUNT += 1

        # # FOR TESTING
        # if COUNT < 0 or COUNT > 0:
            # continue

        # print(f"EXPR: {dirname}")
        pre_args = parse_pretrain_dirname(dirname, preargnamelist)
        pretrain_path = os.path.join(args.dirpath, dirname)
        wrapped_inner = tqdm(next(os.walk(pretrain_path))[1], desc='Unlearn dirs', unit='dir')
        baselines[dirname] = []
        for undirname in wrapped_inner:
            if "(replacement)" in undirname:
                continue
            # print(dirname, undirname)
            un_args = parse_unlearn_dirname(undirname, unargnamelist)
            un_path = os.path.join(pretrain_path, undirname)
            row = parse_unpath(un_path, pre_args, un_args, args, headers)

            if row is None:
                skipped.append((dirname, undirname))
                # print('Skipping', un_path)
                continue
            else:
                row["id"] = dirname
                if row["unlearn_method"] == "Naive" and row["exp_name"] == "unlearn":
                    baselines[dirname].append(row["unlearn_time"])
            rows.append(row)

    for dirname in baselines:
        baselines[dirname] = np.mean(baselines[dirname])
    
    print('Skipped', len(skipped), 'runs')
    for dirname, undirname in skipped:
        print(dirname, undirname)
        pre_args = parse_pretrain_dirname(dirname, preargnamelist)
        un_args = parse_unlearn_dirname(undirname, unargnamelist)
        print(pre_args, un_args)        
    skipped = []

    wrapped_rows = tqdm(range(len(rows)), desc='Calculating costs', unit='row')
    for i in wrapped_rows:
        rows[i]["cost"] = rows[i]["unlearn_time"] / baselines[rows[i]["id"]]

    accumulated_results = {}

    # Build ids for each setting of the unlearn method
    unlearn_ids = []
    for row in rows:
        if row["unlearn_method"] not in unlearn_method:
            continue

        if row["unlearn_method"] == "Naive" and row["exp_name"] != "unlearn":
            continue
        
        # print(row)
        # exit(0)

        id_dict = {}

        for key in unargs_per_method[row["unlearn_method"]]:
            id_dict[key] = row[key]
        id_dict["unlearn_method"] = row["unlearn_method"]

        unlearn_ids.append(id_dict)

    os.makedirs(f"./{args.out_dir}", exist_ok=True)

    chosen_params_reload = {}

    chosen_parameters_per_method = {}

    wrapped_keys = tqdm(baselines.keys(), desc="Iterating over experiments", unit="exp")
    for key in wrapped_keys:
        accumulated_results[key] = {}
        results_to_plot = {}

        key_rows = [row for row in rows if row["id"] == key]
        dataset_method = key_rows[0]["dataset_method"]
        # print(key_rows[0])

        if key not in chosen_parameters_per_method:
            chosen_parameters_per_method[key] = {}

        # Each key is an experiment
        # For each experiment, and for each unlearn method, get the average values and standard deviation across all 10 deletion set sizes
        wrapped_unlearn_ids = tqdm(unlearn_ids, desc="Iterating over grid search", unit="ids")
        for unlearn_id in wrapped_unlearn_ids:
            # print(unlearn_id)
            rows_for_unlearn_id = [row for row in key_rows if all(row[k] == unlearn_id[k] for k in unlearn_id.keys()) and float(row["deletion_size"]) > 0]
            unlearn_method = unlearn_id["unlearn_method"]
            # print(len(rows_for_unlearn_id), unlearn_method)
            # exit(0)
            if len(rows_for_unlearn_id) == 0:
                skipped.append(f"Skipping {key} {unlearn_method}, no runs")
                continue

            # Get the average values and standard deviation across all 10 deletion set sizes
            avg_row = {}
            for metric in metricslist:
                use_list = [row[metric] for row in rows_for_unlearn_id if row[metric] != '']
                if len(use_list) == 0:
                    skipped.append(f"Skipping {key} {unlearn_method}, no values for {metric}")
                    continue
                avg_row[metric] = np.mean(use_list)
                std_row = np.std(use_list)
                avg_row[metric + "_std"] = std_row
            avg_row["id"] = key
            avg_row["unlearn_method"] = unlearn_method
            avg_row["dataset"] = rows_for_unlearn_id[0]["dataset"]
            avg_row["model"] = rows_for_unlearn_id[0]["model"]
            avg_row["dataset_method"] = rows_for_unlearn_id[0]["dataset_method"]
            avg_row["forget_set_size"] = rows_for_unlearn_id[0]["forget_set_size"]
            avg_row["len"] = len(rows_for_unlearn_id)

            if unlearn_method in accumulated_results[key]:
                # print("Already exists", unlearn_method)
                # If the unlearn method already exists, check if the new values are better
                new_run_weighted = avg_row[method_to_retain_measure[dataset_method]] + avg_row["manip_acc"]
                existing_run_weighted = accumulated_results[key][unlearn_method][method_to_retain_measure[dataset_method]] + accumulated_results[key][unlearn_method]["manip_acc"]
                if new_run_weighted > existing_run_weighted or len(rows_for_unlearn_id) > accumulated_results[key][unlearn_method]["len"]:
                # if new_run_weighted > existing_run_weighted:
                    print("Updating", unlearn_method)
                    for metric in accumulated_results[key][unlearn_method].keys():
                        accumulated_results[key][unlearn_method][metric] = avg_row[metric]

                    # Plot the test_retain_acc against the deletion size / forget set size
                    test_retain_acc = [row[method_to_retain_measure[dataset_method]] for row in rows_for_unlearn_id if row[method_to_retain_measure[dataset_method]] != '']
                    manip_acc = [row["manip_acc"] for row in rows_for_unlearn_id if row["manip_acc"] != '']
                    deletion_size = [row["deletion_size"] for row in rows_for_unlearn_id if row["deletion_size"] != '']
                    forget_set_size = float(rows_for_unlearn_id[0]["forget_set_size"])
                    portions = [float(deletion_size[i]) / forget_set_size for i in range(len(deletion_size))]

                    # Reorder all of the results so portions is in increasing order
                    sorted_indices = np.argsort(portions)
                    test_retain_acc = [100 * test_retain_acc[i] for i in sorted_indices]
                    manip_acc = [100 * manip_acc[i] for i in sorted_indices]
                    deletion_size = [deletion_size[i] for i in sorted_indices]
                    forget_set_size = float(rows_for_unlearn_id[0]["forget_set_size"])
                    portions = [portions[i] for i in sorted_indices]

                    if unlearn_method == "Reload":
                        unlearn_method = "Reload"
                        chosen_params_reload[key] = unlearn_id
                    if unlearn_method == "Naive":
                        unlearn_method = "RewoD"

                    chosen_parameters_per_method[key][unlearn_method] = unlearn_id

                    results_to_plot[unlearn_method] = {
                        "test_retain_acc": test_retain_acc,
                        "manip_acc": manip_acc,
                        "deletion_size": deletion_size,
                        "forget_set_size": forget_set_size,
                        "portions": portions,
                    }
            else:
                accumulated_results[key][unlearn_method] = avg_row

                # Plot the test_retain_acc against the deletion size / forget set size
                test_retain_acc = [row[method_to_retain_measure[dataset_method]] for row in rows_for_unlearn_id if row[method_to_retain_measure[dataset_method]] != '']
                manip_acc = [row["manip_acc"] for row in rows_for_unlearn_id if row["manip_acc"] != '']
                deletion_size = [row["deletion_size"] for row in rows_for_unlearn_id if row["deletion_size"] != '']
                forget_set_size = float(rows_for_unlearn_id[0]["forget_set_size"])
                portions = [float(deletion_size[i]) / forget_set_size for i in range(len(deletion_size))]

                sorted_indices = np.argsort(portions)
                test_retain_acc = [100 * test_retain_acc[i] for i in sorted_indices]
                manip_acc = [100 * manip_acc[i] for i in sorted_indices]
                deletion_size = [deletion_size[i] for i in sorted_indices]
                forget_set_size = float(rows_for_unlearn_id[0]["forget_set_size"])
                portions = [portions[i] for i in sorted_indices]

                if unlearn_method == "Reload":
                        unlearn_method = "Reload"
                        chosen_params_reload[key] = unlearn_id
                if unlearn_method == "Naive":
                    unlearn_method = "RewoD"

                    # print("Adding", unlearn_method)
                    # print("test_retain_acc", test_retain_acc)
                    # print("manip_acc", manip_acc)
                    # print("deletion_size", deletion_size)
                    # print("forget_set_size", forget_set_size)
                    # print("portions", portions)
                    # print(rows_for_unlearn_id)
                    # exit(0)

                chosen_parameters_per_method[key][unlearn_method] = unlearn_id

                results_to_plot[unlearn_method] = {
                    "test_retain_acc": test_retain_acc,
                    "manip_acc": manip_acc,
                    "deletion_size": deletion_size,
                    "forget_set_size": forget_set_size,
                    "portions": portions,
                }

        # Plot the test_retain_acc of all unlearn methods for this key
        fig = plt.figure()
        plt.xlabel(r'Identified Fraction $\gamma$')
        plt.ylabel(r'$Acc_{new}$')
        # plt.title(r'$Acc_{new}$ vs Identified Fraction $\gamma$')   
        for unlearn_method in results_to_plot:
            plt.plot(results_to_plot[unlearn_method]["portions"], results_to_plot[unlearn_method]["test_retain_acc"], label=unlearn_method, marker=method_to_markers[unlearn_method], color=method_to_colours[unlearn_method])
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(args.out_dir, f"{key}_test_retain_acc.png"), dpi=400, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        plt.xlabel(r'Identified Fraction $\gamma$')
        plt.ylabel(r'$Acc_{corr}$')
        # plt.title(r'$Acc_{corr}$ vs Identified Fraction $\gamma$') 
        for unlearn_method in results_to_plot:
            plt.plot(results_to_plot[unlearn_method]["portions"], results_to_plot[unlearn_method]["manip_acc"], label=unlearn_method, marker=method_to_markers[unlearn_method], color=method_to_colours[unlearn_method])
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(args.out_dir, f"{key}_corr_acc.png"), dpi=400, bbox_inches='tight')
        plt.close(fig)

    # Report all the baseline results
    pretrain_rows = [row for row in rows if row["unlearn_method"] == "Naive" and row["exp_name"] == "pretrainmodel"]
    pretrained_results = {}
    for row in pretrain_rows:
        pretrained_results[row["id"]] = {
            "retain_acc": row[method_to_retain_measure[row["dataset_method"]]],
            "correct_acc": row["manip_acc"],
            "unlearn_time": row["unlearn_time"],
        }

    df = pd.DataFrame.from_records(pretrained_results)
    df.to_csv(os.path.join(args.out_dir, "pretrained_results.tsv"), sep='\t')

    for key in accumulated_results:
        for method in accumulated_results[key]:
            accumulated_results[key][method]["retain_acc"] = 100 * (accumulated_results[key][method][method_to_retain_measure[accumulated_results[key][method]["dataset_method"]]] - \
                pretrained_results[key]["retain_acc"])
            accumulated_results[key][method]["retain_acc_std"] = 100 * (accumulated_results[key][method][method_to_retain_measure[accumulated_results[key][method]["dataset_method"]] + "_std"])

    # Save the results to a tsv file      
    for key in accumulated_results:
        accumulated_results[key]["id"] = key
        df = pd.DataFrame.from_records(accumulated_results[key])   
        df.to_csv(os.path.join(args.out_dir, f"{key}_accumulated_results.tsv"), sep='\t')

    # FORMAT TO LATEX
    row_format = "{method}   & {retain_c10:.2f} $<CURLOPEN>\pm$ {retain_c10_std:.2f}<CURLCLOSE>  & {retain_c100:.2f} $<CURLOPEN>\pm$ {retain_c100_std:.2f}<CURLCLOSE> \\\\"

    unlearn_method_table_order = ['BadT', 'CF', 'SSD', 'Scrub', 'Naive', 'Reload'] # should be included but its not done yet

    small_poison = [("CIFAR10", "poisoning", "100"), ("CIFAR100", "poisoning", "100")]
    small_ic = [("CIFAR10", "interclasslabelswap", "500"), ("CIFAR100", "interclasslabelswap", "100")]
    mid_poison = [("CIFAR10", "poisoning", "500"), ("CIFAR100", "poisoning", "500")]
    mid_ic = [("CIFAR10", "interclasslabelswap", "2500"), ("CIFAR100", "interclasslabelswap", "250")]
    large_poison = [("CIFAR10", "poisoning", "1000"), ("CIFAR100", "poisoning", "1000")]
    large_ic = [("CIFAR10", "interclasslabelswap", "5000"), ("CIFAR100", "interclasslabelswap", "500")]

    small_poison_dict = {}
    small_ic_dict = {}
    mid_poison_dict = {}
    mid_ic_dict = {}
    large_poison_dict = {}
    large_ic_dict = {}
    
    small_poison_gen = []
    small_ic_gen = []
    mid_poison_gen = []
    mid_ic_gen = []
    large_poison_gen = []
    large_ic_gen = []

    method_to_result = {}

    for key in accumulated_results:
        split_key = key.split('_')

        if split_key[2] == "poisoning" and split_key[3] == "100":
            small_poison_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "poisoning" and split_key[3] == "500":
            mid_poison_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "poisoning" and split_key[3] == "1000":
            large_poison_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "interclasslabelswap" and ((split_key[0] == "CIFAR10" and split_key[3] == "500") or (split_key[0] == "CIFAR100" and split_key[3] == "100")):
            small_ic_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "interclasslabelswap" and ((split_key[0] == "CIFAR10" and split_key[3] == "2500") or (split_key[0] == "CIFAR100" and split_key[3] == "250")):
            mid_ic_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "interclasslabelswap" and ((split_key[0] == "CIFAR10" and split_key[3] == "5000") or (split_key[0] == "CIFAR100" and split_key[3] == "500")):
            large_ic_dict[split_key[0]] = accumulated_results[key]
    
    for method in unlearn_method_table_order:
        method_name = method
        if method == "Reload":
            method_name = '\\' + 'textsc{RELOAD }'
        elif method == "Naive":
            method_name = 'RewoD'
        small_poison_gen.append(row_format.format(method=method_name, retain_c10=small_poison_dict["CIFAR10"][method]["retain_acc"],
                                                  retain_c10_std=small_poison_dict["CIFAR10"][method]["retain_acc_std"], 
                                                  retain_c100=small_poison_dict["CIFAR100"][method]["retain_acc"],
                                                  retain_c100_std=small_poison_dict["CIFAR100"][method]["retain_acc_std"]))
        small_ic_gen.append(row_format.format(method=method_name, retain_c10=small_ic_dict["CIFAR10"][method]["retain_acc"],
                                                    retain_c10_std=small_ic_dict["CIFAR10"][method]["retain_acc_std"], 
                                                    retain_c100=small_ic_dict["CIFAR100"][method]["retain_acc"],
                                                    retain_c100_std=small_ic_dict["CIFAR100"][method]["retain_acc_std"]))
        mid_poison_gen.append(row_format.format(method=method_name, retain_c10=mid_poison_dict["CIFAR10"][method]["retain_acc"],
                                                    retain_c10_std=mid_poison_dict["CIFAR10"][method]["retain_acc_std"], 
                                                    retain_c100=mid_poison_dict["CIFAR100"][method]["retain_acc"],
                                                    retain_c100_std=mid_poison_dict["CIFAR100"][method]["retain_acc_std"]))
        mid_ic_gen.append(row_format.format(method=method_name, retain_c10=mid_ic_dict["CIFAR10"][method]["retain_acc"],
                                                    retain_c10_std=mid_ic_dict["CIFAR10"][method]["retain_acc_std"], 
                                                    retain_c100=mid_ic_dict["CIFAR100"][method]["retain_acc"],
                                                    retain_c100_std=mid_ic_dict["CIFAR100"][method]["retain_acc_std"]))
        large_poison_gen.append(row_format.format(method=method_name, retain_c10=large_poison_dict["CIFAR10"][method]["retain_acc"],
                                                    retain_c10_std=large_poison_dict["CIFAR10"][method]["retain_acc_std"], 
                                                    retain_c100=large_poison_dict["CIFAR100"][method]["retain_acc"],
                                                    retain_c100_std=large_poison_dict["CIFAR100"][method]["retain_acc_std"]))   
        large_ic_gen.append(row_format.format(method=method_name, retain_c10=large_ic_dict["CIFAR10"][method]["retain_acc"],
                                                    retain_c10_std=large_ic_dict["CIFAR10"][method]["retain_acc_std"], 
                                                    retain_c100=large_ic_dict["CIFAR100"][method]["retain_acc"],
                                                    retain_c100_std=large_ic_dict["CIFAR100"][method]["retain_acc_std"]))
    
    for i in range(len(small_poison_gen)):
        small_poison_gen[i] = small_poison_gen[i].replace("<CURLOPEN>", "_{")
        small_poison_gen[i] = small_poison_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(small_ic_gen)):
        small_ic_gen[i] = small_ic_gen[i].replace("<CURLOPEN>", "_{")
        small_ic_gen[i] = small_ic_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(mid_poison_gen)):
        mid_poison_gen[i] = mid_poison_gen[i].replace("<CURLOPEN>", "_{")
        mid_poison_gen[i] = mid_poison_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(mid_ic_gen)):
        mid_ic_gen[i] = mid_ic_gen[i].replace("<CURLOPEN>", "_{")
        mid_ic_gen[i] = mid_ic_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(large_poison_gen)):
        large_poison_gen[i] = large_poison_gen[i].replace("<CURLOPEN>", "_{")
        large_poison_gen[i] = large_poison_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(large_ic_gen)):
        large_ic_gen[i] = large_ic_gen[i].replace("<CURLOPEN>", "_{")
        large_ic_gen[i] = large_ic_gen[i].replace("<CURLCLOSE>", "}")

    # Save the results to a text
    with open(os.path.join(args.out_dir, "small_poison_retain_acc.txt"), 'w') as f:
        for line in small_poison_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "small_ic_retain_acc.txt"), 'w') as f:
        for line in small_ic_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "mid_poison_retain_acc.txt"), 'w') as f:
        for line in mid_poison_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "mid_ic_retain_acc.txt"), 'w') as f:
        for line in mid_ic_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "large_poison_retain_acc.txt"), 'w') as f:
        for line in large_poison_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "large_ic_retain_acc.txt"), 'w') as f:
        for line in large_ic_gen:
            f.write(line + '\n')

    small_poison_dict = {}
    small_ic_dict = {}
    mid_poison_dict = {}
    mid_ic_dict = {}
    large_poison_dict = {}
    large_ic_dict = {}
    
    small_poison_gen = []
    small_ic_gen = []
    mid_poison_gen = []
    mid_ic_gen = []
    large_poison_gen = []
    large_ic_gen = []

    method_to_result = {}

    for key in accumulated_results:
        split_key = key.split('_')

        if split_key[2] == "poisoning" and split_key[3] == "100":
            small_poison_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "poisoning" and split_key[3] == "500":
            mid_poison_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "poisoning" and split_key[3] == "1000":
            large_poison_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "interclasslabelswap" and ((split_key[0] == "CIFAR10" and split_key[3] == "500") or (split_key[0] == "CIFAR100" and split_key[3] == "100")):
            small_ic_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "interclasslabelswap" and ((split_key[0] == "CIFAR10" and split_key[3] == "2500") or (split_key[0] == "CIFAR100" and split_key[3] == "250")):
            mid_ic_dict[split_key[0]] = accumulated_results[key]
        if split_key[2] == "interclasslabelswap" and ((split_key[0] == "CIFAR10" and split_key[3] == "5000") or (split_key[0] == "CIFAR100" and split_key[3] == "500")):
            large_ic_dict[split_key[0]] = accumulated_results[key]
    
    for method in unlearn_method_table_order:
        method_name = method
        if method == "Reload":
            method_name = '\\' + 'textsc{RELOAD (ours)}'
        elif method == "Naive":
            method_name = 'RewoD'
        small_poison_gen.append(row_format.format(method=method_name, retain_c10=small_poison_dict["CIFAR10"][method]["cost"],
                                                  retain_c10_std=small_poison_dict["CIFAR10"][method]["cost_std"], 
                                                  retain_c100=small_poison_dict["CIFAR100"][method]["cost"],
                                                  retain_c100_std=small_poison_dict["CIFAR100"][method]["cost_std"]))
        small_ic_gen.append(row_format.format(method=method_name, retain_c10=small_ic_dict["CIFAR10"][method]["cost"],
                                                    retain_c10_std=small_ic_dict["CIFAR10"][method]["cost_std"], 
                                                    retain_c100=small_ic_dict["CIFAR100"][method]["cost"],
                                                    retain_c100_std=small_ic_dict["CIFAR100"][method]["cost_std"]))
        mid_poison_gen.append(row_format.format(method=method_name, retain_c10=mid_poison_dict["CIFAR10"][method]["cost"],
                                                    retain_c10_std=mid_poison_dict["CIFAR10"][method]["cost_std"], 
                                                    retain_c100=mid_poison_dict["CIFAR100"][method]["cost"],
                                                    retain_c100_std=mid_poison_dict["CIFAR100"][method]["cost_std"]))
        mid_ic_gen.append(row_format.format(method=method_name, retain_c10=mid_ic_dict["CIFAR10"][method]["cost"],
                                                    retain_c10_std=mid_ic_dict["CIFAR10"][method]["cost_std"], 
                                                    retain_c100=mid_ic_dict["CIFAR100"][method]["cost"],
                                                    retain_c100_std=mid_ic_dict["CIFAR100"][method]["cost_std"]))
        large_poison_gen.append(row_format.format(method=method_name, retain_c10=large_poison_dict["CIFAR10"][method]["cost"],
                                                    retain_c10_std=large_poison_dict["CIFAR10"][method]["cost_std"], 
                                                    retain_c100=large_poison_dict["CIFAR100"][method]["cost"],
                                                    retain_c100_std=large_poison_dict["CIFAR100"][method]["cost_std"]))   
        large_ic_gen.append(row_format.format(method=method_name, retain_c10=large_ic_dict["CIFAR10"][method]["cost"],
                                                    retain_c10_std=large_ic_dict["CIFAR10"][method]["cost_std"], 
                                                    retain_c100=large_ic_dict["CIFAR100"][method]["cost"],
                                                    retain_c100_std=large_ic_dict["CIFAR100"][method]["cost_std"]))
    
    for i in range(len(small_poison_gen)):
        small_poison_gen[i] = small_poison_gen[i].replace("<CURLOPEN>", "_{")
        small_poison_gen[i] = small_poison_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(small_ic_gen)):
        small_ic_gen[i] = small_ic_gen[i].replace("<CURLOPEN>", "_{")
        small_ic_gen[i] = small_ic_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(mid_poison_gen)):
        mid_poison_gen[i] = mid_poison_gen[i].replace("<CURLOPEN>", "_{")
        mid_poison_gen[i] = mid_poison_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(mid_ic_gen)):
        mid_ic_gen[i] = mid_ic_gen[i].replace("<CURLOPEN>", "_{")
        mid_ic_gen[i] = mid_ic_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(large_poison_gen)):
        large_poison_gen[i] = large_poison_gen[i].replace("<CURLOPEN>", "_{")
        large_poison_gen[i] = large_poison_gen[i].replace("<CURLCLOSE>", "}")
    for i in range(len(large_ic_gen)):
        large_ic_gen[i] = large_ic_gen[i].replace("<CURLOPEN>", "_{")
        large_ic_gen[i] = large_ic_gen[i].replace("<CURLCLOSE>", "}")

    # Save the results to a text
    with open(os.path.join(args.out_dir, "small_poison_cost.txt"), 'w') as f:
        for line in small_poison_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "small_ic_cost.txt"), 'w') as f:
        for line in small_ic_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "mid_poison_cost.txt"), 'w') as f:
        for line in mid_poison_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "mid_ic_cost.txt"), 'w') as f:
        for line in mid_ic_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "large_poison_cost.txt"), 'w') as f:
        for line in large_poison_gen:
            f.write(line + '\n')
    with open(os.path.join(args.out_dir, "large_ic_cost.txt"), 'w') as f:
        for line in large_ic_gen:
            f.write(line + '\n')

    # Save chosen reload params to a json
    with open(os.path.join(args.out_dir, "chosen_params_reload.json"), 'w') as f:
        json.dump(chosen_params_reload, f, indent=4)

    with open(os.path.join(args.out_dir, "chosen_params.json"), 'w') as f:
        json.dump(chosen_parameters_per_method, f, indent=4)

    df = pd.DataFrame.from_records(rows)
    df.to_csv(os.path.join(args.out_dir, args.tsvpath), sep='\t')

    print(f"Skipped {len(skipped)} runs for plotting")
    for st in skipped:
        print(st)

    