from argparse import ArgumentParser
from model import initialize_model_and_tokenizer, DecoderLensWrapper
from sentence_transformers import SentenceTransformer
from dataset import load_qa_dataset_for_inference, load_nli_dataset_for_inference
import gc
import torch
from eval import evaluate_sim, compute_query_matrix_and_norm, evaluate_f1_max, evaluate_exact_max, compute_acc_nli
from tqdm import tqdm
import pdb

def main(args):
    #initialize
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    decoder_lens_model = DecoderLensWrapper(model, tokenizer)
    
    

    if args.task_type == 'qa':
        queries_dl, same_lang_answers, cross_lang_answers, neg_same_lang_answers, neg_cross_lang_answers, hard_neg_same_lang_answers, hard_neg_cross_lang_answers, _ = load_qa_dataset_for_inference(args.dataset_file, args.batch_size)
        source_preds_per_layer = []
        target_preds_per_layer = []
        source_last_layer_preds = []
        target_last_layer_preds = []
        # use ordinary inference
        if -1 in args.probed_layers:
            source_preds, target_preds = decoder_lens_model.decode(queries_dl)
            source_preds_per_layer.append(source_preds)
            target_preds_per_layer.append(target_preds)
            
        # probe particular layer
        else:
            # get all inputs required for decoder
            enc_hidden_states, attention_masks, tgt_batches = decoder_lens_model.get_dec_inputs(queries_dl)
            for probed_layer in args.probed_layers:
                source_preds = decoder_lens_model.decode_on_particular_hidden_state(enc_hidden_states, attention_masks, tgt_batches, probed_layer)
                source_preds_per_layer.append(source_preds)
                target_preds_per_layer.append(target_preds)
            source_last_layer_preds, target_last_layer_preds = decoder_lens_model.decode(queries_dl)

        # clear model to free up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # evaluate
        if args.metrics == 'avg_cos_sim':
            sim_model = SentenceTransformer('sentence-transformers/LaBSE')
        else:
            sim_model = None
        same_lang_scores_avg_per_layer = []
        cross_lang_scores_avg_per_layer = []
        neg_same_lang_scores_avg_per_layer = []
        neg_cross_lang_scores_avg_per_layer = []
        hard_neg_same_lang_scores_avg_per_layer = []
        hard_neg_cross_lang_scores_avg_per_layer = []

        for source_preds, target_preds in zip(source_preds_per_layer, target_preds_per_layer):
            if args.metrics == 'avg_cos_sim':
                source_query_embed, source_query_norm = compute_query_matrix_and_norm(sim_model, source_preds)
                target_query_embed, target_query_norm = compute_query_matrix_and_norm(sim_model, target_preds)

                same_lang_sim_avg = evaluate_sim(sim_model, source_query_embed, source_query_norm, same_lang_answers) # same language answers
                cross_lang_sim_avg = evaluate_sim(sim_model, target_query_embed, target_query_norm, cross_lang_answers) # cross language answers
                neg_same_lang_sim_avg = evaluate_sim(sim_model, source_query_embed, source_query_norm, neg_same_lang_answers) # same language answers
                neg_cross_lang_sim_avg = evaluate_sim(sim_model, target_query_embed, target_query_norm, neg_cross_lang_answers) # cross language answers
                hard_neg_same_lang_sim_avg = evaluate_sim(sim_model, source_query_embed, source_query_norm, hard_neg_same_lang_answers) # same language answers
                hard_neg_cross_lang_sim_avg = evaluate_sim(sim_model, target_query_embed, target_query_norm, hard_neg_cross_lang_answers) # cross language answers
                
                same_lang_scores_avg_per_layer.append(same_lang_sim_avg)
                cross_lang_scores_avg_per_layer.append(cross_lang_sim_avg)
                neg_same_lang_scores_avg_per_layer.append(neg_same_lang_sim_avg)
                neg_cross_lang_scores_avg_per_layer.append(neg_cross_lang_sim_avg)
                hard_neg_same_lang_scores_avg_per_layer.append(hard_neg_same_lang_sim_avg)
                hard_neg_cross_lang_scores_avg_per_layer.append(hard_neg_cross_lang_sim_avg)
            
            elif args.metrics == 'f1_max':
                cross_lang_scores_avg_per_layer.append(evaluate_f1_max(target_preds, cross_lang_answers, args.target_lang))
                neg_cross_lang_scores_avg_per_layer.append(evaluate_f1_max(target_preds, neg_cross_lang_answers, args.target_lang))
                hard_neg_cross_lang_scores_avg_per_layer.append(evaluate_f1_max(target_preds, hard_neg_cross_lang_answers, args.target_lang))

                same_lang_scores_avg_per_layer.append(evaluate_f1_max(source_preds, same_lang_answers, args.source_lang))
                neg_same_lang_scores_avg_per_layer.append(evaluate_f1_max(source_preds, neg_same_lang_answers, args.source_lang))
                hard_neg_same_lang_scores_avg_per_layer.append(evaluate_f1_max(source_preds, hard_neg_same_lang_answers, args.source_lang))
            
            elif args.metrics == 'exact_max':
                cross_lang_scores_avg_per_layer.append(evaluate_exact_max(target_preds, cross_lang_answers, args.target_lang))
                neg_cross_lang_scores_avg_per_layer.append(evaluate_exact_max(target_preds, neg_cross_lang_answers, args.target_lang))
                hard_neg_cross_lang_scores_avg_per_layer.append(evaluate_exact_max(target_preds, hard_neg_cross_lang_answers, args.target_lang))

                same_lang_scores_avg_per_layer.append(evaluate_exact_max(source_preds, same_lang_answers, args.source_lang))
                neg_same_lang_scores_avg_per_layer.append(evaluate_exact_max(source_preds, neg_same_lang_answers, args.source_lang))
                hard_neg_cross_lang_scores_avg_per_layer.append(evaluate_exact_max(source_preds, hard_neg_same_lang_answers, args.source_lang))

        
        print(f"Source language: {args.source_lang}")
        print(f"Target language: {args.target_lang}")
        print()
        print()

        if len(source_last_layer_preds) > 0:
            overall_same_lang_scores = -1000
            overall_neg_same_lang_scores = -1000
            overall_hard_neg_lang_scores = -1000
            if args.metrics == 'avg_cos_sim':
                source_query_embed, source_query_norm = compute_query_matrix_and_norm(sim_model, source_last_layer_preds)
                target_query_embed, target_query_norm = compute_query_matrix_and_norm(sim_model, target_last_layer_preds)

                overall_same_lang_scores = evaluate_sim(sim_model,  source_query_embed, source_query_norm, same_lang_answers)
                overall_cross_lang_scores = evaluate_sim(sim_model, target_query_embed, target_query_norm, cross_lang_answers)
                overall_neg_same_lang_scores = evaluate_sim(sim_model, source_query_embed, source_query_norm, neg_same_lang_answers)
                overall_neg_cross_lang_scores = evaluate_sim(sim_model, target_query_embed, target_query_norm, neg_cross_lang_answers)
                overall_hard_neg_same_lang_scores = evaluate_sim(sim_model, source_query_embed, source_query_norm, hard_neg_same_lang_answers)
                overall_hard_neg_cross_lang_scores = evaluate_sim(sim_model, target_query_embed, target_query_norm, hard_neg_cross_lang_answers)
            
            elif args.metrics == 'f1_max':
                overall_cross_lang_scores = evaluate_f1_max(target_preds, cross_lang_answers, args.target_lang)
                overall_neg_cross_lang_scores = evaluate_f1_max(target_preds, neg_cross_lang_answers, args.target_lang)
                overall_hard_neg_cross_lang_scores = evaluate_f1_max(target_preds, hard_neg_cross_lang_answers, args.target_lang)

                overall_same_lang_scores = evaluate_f1_max(source_preds, same_lang_answers, args.source_lang)
                overall_neg_same_lang_scores = evaluate_f1_max(source_preds, neg_same_lang_answers, args.source_lang)
                overall_hard_neg_same_lang_scores = evaluate_f1_max(source_preds, hard_neg_same_lang_answers, args.source_lang)

            elif args.metrics == 'exact_max':
                overall_cross_lang_scores = evaluate_exact_max(target_preds, cross_lang_answers, args.target_lang)
                overall_neg_cross_lang_scores = evaluate_exact_max(target_preds, neg_cross_lang_answers, args.target_lang)
                overall_hard_neg_cross_lang_scores = evaluate_exact_max(target_preds, hard_neg_cross_lang_answers, args.target_lang)

                overall_same_lang_scores = evaluate_exact_max(source_preds, same_lang_answers, args.source_lang)
                overall_neg_same_lang_scores = evaluate_exact_max(source_preds, neg_same_lang_answers, args.source_lang)
                overall_hard_neg_same_lang_scores = evaluate_exact_max(source_preds, hard_neg_same_lang_answers, args.source_lang)
                

            print(f"Overall same language scores: {overall_same_lang_scores}")
            print(f"Overall cross language scores: {overall_cross_lang_scores}")
            print()
            print(f"Overall negative same language scores: {overall_neg_same_lang_scores}")
            print(f"Overall negative cross language scores: {overall_neg_cross_lang_scores}")
            print()
            print(f"Overall hard negative same language scores: {overall_hard_neg_same_lang_scores}")
            print(f"Overall hard negative cross language scores: {overall_hard_neg_cross_lang_scores}")
            print()
            print()


        print(f"Positive same language similarity average: {same_lang_scores_avg_per_layer}")
        print(f"Positive cross language similarity average: {cross_lang_scores_avg_per_layer}")
        print()
        print(f"Negative same language similarity average: {neg_same_lang_scores_avg_per_layer}")
        print(f"Negative cross language similarity average: {neg_cross_lang_scores_avg_per_layer}")
        print()
        print(f"Hard negative same language similarity average: {hard_neg_same_lang_scores_avg_per_layer}")
        print(f"Hard negative cross language similarity average: {hard_neg_cross_lang_scores_avg_per_layer}")
    
    # nli_qa
    elif args.task_type == 'nli_qa':
        nli_pairs, _ = load_nli_dataset_for_inference(args.dataset_file)
        preds, gts = decoder_lens_model.classify(nli_pairs, args.nli_labels, True)
        accuracy = compute_acc_nli(preds, gts)
        print(f"Accuracy: {accuracy}")
        
    # nli
    else:
        nli_pairs, _ = load_nli_dataset_for_inference(args.dataset_file)
        preds, gts = decoder_lens_model.classify(nli_pairs, args.nli_labels, False)
        accuracy = compute_acc_nli(preds, gts)
        print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_type', type=str, default='qa', choices=['nli', 'qa', 'nli_qa'])
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--probed_layers', type=int, nargs='+')
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--metrics', type=str, default='avg_cos_sim', choices=['avg_cos_sim','f1_max', 'exact_max'])
    parser.add_argument('--nli_labels', required=False, type=str, nargs='+')


    args = parser.parse_args()

    main(args)