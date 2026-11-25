import logging
import os
import pathlib

import sys
import torch
import copy

from config_files.prune_configs import prune_exper
from general_utils import data_utils, gpu_utils, hf_utils
from pruning_methods.wanda.prune_wanda import prune_wanda
from pruning_methods.sparsegpt.prune_sparsegpt import prune_sparsegpt
from pruning_methods.OATS.prune_oats import prune_oats
from pruning_methods.OATS.prune_oats_compress import prune_oats_compress
from pruning_methods.DSnoT.prune_dsnot import prune_DSnoT
from pruning_methods.OATS.oats_utils import load_oats
from general_utils.config import config
from measure_utils.measure_utils import DumpJSON
from ml_collections import ConfigDict
import lm_eval
from lm_eval.models.huggingface import HFLM
from general_utils.utils import calculate_avg_accuracy
from pruning_methods.pruning_utils import calc_outlier_ratio

def process_pruning_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity <= 1:
        raise ValueError

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "bf16":
        config.dtype = torch.bfloat16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise ValueError


def pruning_main(args, checkpoint_path, results_path) -> None:

    results = DumpJSON(read_path=(results_path+'.json'),
                    write_path=(results_path+'.json'))
    
    results_stats = {
                'target_sparsity': args.sparsity,
                'model'    : args.model,
                'prune_type': args.prune_type,
                }

    for k,v in args.prune_hyper.items():
        results_stats[k] = args.prune_hyper[k]
    
    presampled_data_path = args.get("presampled_data_path", None)
    presampled_tuple_keys = args.get("presampled_tuple_keys", "auto")

    if presampled_data_path:
        train_dataset = None
        print(f"Using presampled calibration data from: {presampled_data_path}", flush=True)
    else:
        train_dataset = data_utils.get_dataset(args.cal_dataset, train=True)
        print("Finished Train Set", flush=True)

    # Checkpointing for pruning
    if os.path.exists(checkpoint_path + "/prune_chkpt.pt"):
        if args.prune_type == "OATS" and args.prune_hyper['compress']:
            model_adapter, tokenizer = load_oats(
                    args.model,
                    args.sparsity,
                    args.prune_hyper,
                    checkpoint_path,
                    dtype=config.dtype
                )
        else:
            model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
                args.model,
                checkpoint_path,
                dtype=config.dtype
            )
    else:
         # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, None, dtype=config.dtype
        )

    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    print("Start loading data", flush=True)

    if presampled_data_path:
        train_loader = data_utils.prepare_presampled_dataloader(
            data_path=presampled_data_path,
            batch_size=args.cal_batch_size,
            tuple_keys=presampled_tuple_keys,
        )
    else:
        train_loader = data_utils.prepare_dataloader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            max_seqlen=args.cal_max_seqlen,
            batch_size=args.cal_batch_size,
            nsamples=args.cal_nsamples,
            varied_seqlen=args.varied_seqlen,
            seed=args.seed,
        )

    print("Finished Calibration Loader", flush=True)
    
    original_param_count = sum(p.numel() for p in model.parameters())
    print(f'Original model parameters: {original_param_count:,d}')

    # ========================= Pruning Code ========================================
    # OWL Sparsity ratios
    if args.use_owl and not os.path.exists(checkpoint_path + "/prune_chkpt.pt"):
        if args.model == "llama3-8b":
            owl_deltas = [0.030332429999999966, -0.02647441000000006, 0.06844161999999998, \
             0.08346081999999999, 0.03263278999999997, 0.037657790000000024, \
                0.03329596000000001, 0.03576663999999996, 0.036910620000000005, \
                    0.035973359999999954, 0.040336619999999934, 0.03760768000000003, \
                        0.025321199999999933, 0.04359360999999995, 0.029471359999999946, \
                            0.02946841999999994, 0.005488669999999973, 0.007622550000000006, \
                                0.0034176900000000288, -0.004022390000000042, -0.013647900000000046, \
                                    -0.028375209999999984, -0.03668506000000005, -0.03929044000000004, \
                                        -0.04648211000000002, -0.049364460000000054, -0.05529379000000001, -0.05152182000000005, \
                                            -0.06396902999999998, -0.06046841999999997, -0.06466561000000004, -0.07653918000000004]
            layerwise_sparsity_ratios = [args.sparsity - z for z in owl_deltas]
        elif args.model == "phi-3-mini":
            owl_deltas = [0.08126813, 0.05297704999999997, 0.00041484999999996663, \
                          0.006921349999999937, 0.08665363999999998, 0.07546628, \
                            0.07553316999999993, 0.07855594999999993, 0.056884859999999926, \
                                0.038578749999999995, 0.049824209999999924, 0.040341720000000025, \
                                    0.03991473999999995, 0.04626500999999994, 0.013145949999999962, \
                                        0.014221999999999957, 0.002132220000000018, -0.016952010000000017,\
                                              -0.030284560000000016, -0.03848494000000002, -0.04947486000000001,\
                                                  -0.05100441, -0.061722900000000025, -0.06528701000000003, \
                                                    -0.06941421999999997, -0.06816302000000007, -0.07235778999999998, \
                                                        -0.07334636000000005, -0.06708155999999998, -0.05653567999999998, \
                                                            -0.04356389999999999, 0.004573309999999942]
            layerwise_sparsity_ratios = [args.sparsity - z for z in owl_deltas]
        elif args.model == "phi-3-medium":
            owl_deltas = [0.14069135, 0.025625110000000006, 0.009886300000000015, 0.004827680000000001,\
                           0.007173959999999924, -0.00207728000000007, 0.02972733999999999, 0.007524179999999991, \
                            0.010725819999999997, 0.014837989999999968, 0.00669527999999997, 0.009311449999999999, \
                                0.005365590000000031, 0.0005600599999999734, 0.00659628000000001, -0.0017683700000000746, \
                                    0.004343019999999975, -0.0006194299999999764, -0.0021560800000000047, -0.005421720000000074, \
                                        -0.00972521000000004, -0.010528179999999998, -0.010683740000000053, -0.013547040000000066, \
                                        -0.013863979999999998, -0.014902909999999991, -0.016656769999999987, -0.01694910000000005, \
                                            -0.017826019999999998, -0.018716930000000076, -0.01930865000000004, -0.019288410000000034, \
                                                -0.019021520000000014, -0.018256620000000057, -0.01754460000000002, -0.016081789999999985, \
                                                    -0.012885279999999999, -0.009782500000000027, -0.00573477, 0.00945547999999996]
            layerwise_sparsity_ratios = [args.sparsity - z for z in owl_deltas]
        else:
            layerwise_sparsity_ratios = calc_outlier_ratio(model_adapter, args.sparsity, train_loader)
    else:
        layerwise_sparsity_ratios = None

    prune_n, prune_m = 0, 0
    if args.prune_hyper['sparsity_type'] != "unstructured":
        assert args.sparsity == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.prune_hyper['sparsity_type'].split(":"))
    
    if args.prune_type == "wanda":
        prune_wanda(model_adapter, tokenizer, args.sparsity, layerwise_sparsity_ratios, train_loader, checkpoint_path, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_type == "sparse_gpt":
        prune_sparsegpt(model_adapter, tokenizer, args.sparsity, layerwise_sparsity_ratios, train_loader, checkpoint_path, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_type == "dsnot":
        prune_DSnoT(model_adapter, tokenizer, args.sparsity, layerwise_sparsity_ratios, train_loader, args.prune_hyper, checkpoint_path, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_type == "OATS":
        if args.prune_hyper['compress']:
            prune_oats_compress(model_adapter, args.sparsity, layerwise_sparsity_ratios, train_loader, args.prune_hyper, checkpoint_path, prune_n=prune_n, prune_m=prune_m)
        else:
            prune_oats(model_adapter, tokenizer, args.sparsity, layerwise_sparsity_ratios, train_loader, args.prune_hyper, checkpoint_path, prune_n=prune_n, prune_m=prune_m)

    pruned_param_count = sum(int(torch.count_nonzero(p).item()) for p in model.parameters())
    pruned_fraction = 1.0 - (float(pruned_param_count) / float(original_param_count))
    results_stats["final_sparsity"] = pruned_fraction
    results_stats["pruned nnz"] = pruned_param_count
    results_stats["original nnz"] = original_param_count
    
    print(f'Pruned model parameters: {pruned_param_count:,d} (Sparsity: {pruned_fraction:.4f})')

    # =================================================

    # Run PPL Eval
    reset_model_device()
    eval_batch_size = "auto"
    
    if args.eval_ppl:
        hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=eval_batch_size) 
        with torch.no_grad():
            ppl_tasks = ["wikitext"]
            ppl_results = lm_eval.simple_evaluate(hflm, tasks=ppl_tasks, num_fewshot=None, batch_size=eval_batch_size)[
                    'results'
                ]
            print(ppl_results)
            
            ppl_vals = {task: round(result.get('word_perplexity,none', result['word_perplexity,none']), 4) for task, result in ppl_results.items()}

            for k, v in ppl_vals.items():
                results_stats["Task Name"] = k 
                results_stats["Task Score"] = v
                results.append(copy.deepcopy(results_stats))
                results.save()
            
            results.to_csv()

    # ============== Run Zeroshot Eval ================
    if args.eval_zero_shot:
        hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=eval_batch_size) 
        with torch.no_grad():
            zero_shot_tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "rte", "openbookqa", "boolq"]

            ### LM Eval Harness ###
            zs_results = lm_eval.simple_evaluate(hflm, tasks=zero_shot_tasks, num_fewshot=0, batch_size=eval_batch_size)[
                'results'
            ]
            print(zs_results)

            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in zs_results.items()}

            acc_avg = calculate_avg_accuracy(zero_shot_tasks, zs_results)

            metric_vals['average_zero_shot'] = round(acc_avg, 4)

            print(metric_vals)
            
            for k, v in metric_vals.items():
                results_stats["Task Name"] = k
                results_stats["Task Score"] = v
                results.append(copy.deepcopy(results_stats))
                results.save()
            
            results.to_csv()
    
    if args.eval_mmlu:
        hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=eval_batch_size) 
        with torch.no_grad():
            print("Evaluating MMLU!")

            mmlu_tasks = ["mmlu_abstract_algebra", "mmlu_business_ethics", "mmlu_college_computer_science", \
                            "mmlu_college_mathematics", "mmlu_conceptual_physics", "mmlu_formal_logic", "mmlu_machine_learning",\
                                "mmlu_miscellaneous", "mmlu_philosophy", "mmlu_global_facts"]
            
            mmlu_results = lm_eval.simple_evaluate(hflm, tasks=mmlu_tasks, num_fewshot=5, batch_size=eval_batch_size)[
                'results'
            ]

            print(mmlu_results)

            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in mmlu_results.items()}

            mmlu_avg = calculate_avg_accuracy(mmlu_tasks, mmlu_results)
            metric_vals['average_mmlu'] = round(mmlu_avg, 4)

            print(metric_vals)
            
            for k, v in metric_vals.items():
                results_stats["Task Name"] = k
                results_stats["Task Score"] = v
                results.append(copy.deepcopy(results_stats))
                results.save()
            
            results.to_csv()

    return

if __name__ == "__main__":

    run_id = int(sys.argv[1])

    exper_config = prune_exper[run_id - 1]

    pruning_args =  ConfigDict(exper_config)
    if len(sys.argv) >= 3 and sys.argv[2].lower() != "none":
        pruning_args.presampled_data_path = sys.argv[2]
    if len(sys.argv) >= 4 and sys.argv[3].lower() != "auto":
        pruning_args.presampled_tuple_keys = sys.argv[3]
    process_pruning_args(pruning_args)

    results_path = "../" # Please fill in the desired directory for saving results

    checkpoint_path = "../" # Please fill in the desired directoy for checkpointing the pruned model

    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    pruning_main(pruning_args, checkpoint_path, results_path)
