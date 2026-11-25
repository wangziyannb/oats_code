### DSnoT Hyperparams ######################################################################
dsnot_configs = []
max_cycle_time_list = [50]
pow_of_var_regrowing_list = [1]
update_threshold_list = [0.1]
without_same_sign_list = [True]
initial_method_list = ["sparsegpt", "wanda"]

for mct_idx in range(len(max_cycle_time_list)):
    for povr_idx in range(len(pow_of_var_regrowing_list)):
        for ut_idx in range(len(update_threshold_list)):
            for wss_idx in range(len(without_same_sign_list)):
                for im_idx in range(len(initial_method_list)):
                    dsnot_config = {
                                    'max_cycle_time': max_cycle_time_list[mct_idx],
                                    'pow_of_var_regrowing': pow_of_var_regrowing_list[povr_idx],
                                    'update_threshold': update_threshold_list[ut_idx], 
                                    'without_same_sign': without_same_sign_list[wss_idx],
                                    'initial_method': initial_method_list[im_idx]
                    }
                    dsnot_configs.append(dsnot_config)

### OATS Hyperparams ######################################################################
oats_configs = []
unscaled_list = [False]
rank_ratio_list = [0.25, 0.3]
num_iters_list = [80]
prune_level_list = ["row"]

for rank_idx in range(len(rank_ratio_list)):
    for u_idx in range(len(unscaled_list)):
        for ni_idx in range(len(num_iters_list)):
            for pl_idx in range(len(prune_level_list)):
                oats_config = {
                                'unscaled': unscaled_list[u_idx],
                                'rank_ratio': rank_ratio_list[rank_idx],
                                'num_iters': num_iters_list[ni_idx], 
                                'compress'   : True,
                                'prune_level': prune_level_list[pl_idx]

                }
                oats_configs.append(oats_config)

model_list = ["phi-3-mini", "phi-3-medium", "llama3-8b"] 
prune_list = ["OATS", "dsnot", "sparse_gpt", "wanda", "dense"]

prune_exper = []
counter = 1
for m_idx in range(len(model_list)):
    for p_idx in range(len(prune_list)):
        if prune_list[p_idx] == "dense":
            sparsity_list = [1.0]
        else:
            sparsity_list = [0.3, 0.4, 0.5, 0.6]
        
        for s_idx in range(len(sparsity_list)):
            
            if sparsity_list[s_idx] > 0.5 and prune_list[p_idx] != "dense":
                use_owl = True
            else:
                use_owl = False
            
            prune_hyper_list = []

            if prune_list[p_idx] == "OATS":
                if sparsity_list[s_idx] == 0.5:
                    sparsity_type_list = ["unstructured", "2:8"]
                else:
                    sparsity_type_list = ["unstructured"]
                for oats_idx in range(len(oats_configs)):
                    for st_idx in range(len(sparsity_type_list)):
                        oats_configs[oats_idx]["sparsity_type"] = sparsity_type_list[st_idx]
                        prune_hyper_list.append(oats_configs[oats_idx].copy())
            
            elif prune_list[p_idx] == "dsnot":
                if sparsity_list[s_idx] == 0.5:
                    sparsity_type_list = ["unstructured", "2:4"]
                else:
                    sparsity_type_list = ["unstructured"]
                for dsnot_idx in range(len(dsnot_configs)):
                    for st_idx in range(len(sparsity_type_list)):
                        dsnot_configs[dsnot_idx]["sparsity_type"] = sparsity_type_list[st_idx]
                        prune_hyper_list.append(dsnot_configs[dsnot_idx].copy())
                    
            else:
                if sparsity_list[s_idx] == 0.5:
                    sparsity_type_list = ["unstructured", "2:4"]
                else:
                    sparsity_type_list = ["unstructured"]

                for st_idx in range(len(sparsity_type_list)):
                    prune_hyper_list.append ({
                        "sparsity_type": sparsity_type_list[st_idx]
                    })
            
            for ph_idx in range(len(prune_hyper_list)):
                exper_dict = {
                    "Experiment Number": counter,
                    'model': model_list[m_idx],
                    'prune_type': prune_list[p_idx],
                    'sparsity': sparsity_list[s_idx],
                    'dtype': "bf16",
                    "distribute_model": True,
                    "device": None,
                    "cal_dataset": "c4",
                    "cal_nsamples": 128,
                    "cal_batch_size": 32,
                    "cal_max_seqlen": 2048,
                    "varied_seqlen": False,
                    'use_owl': use_owl, 
                    "seed": 42,
                    "save_model": False,
                    "eval_zero_shot": True,
                    "eval_mmlu": True,
                    "eval_ppl": True, 
                }
                exper_dict['prune_hyper'] = prune_hyper_list[ph_idx]
                prune_exper.append(exper_dict)
                #print(exper_dict)
                counter += 1