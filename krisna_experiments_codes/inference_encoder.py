from argparse import ArgumentParser
from model import initialize_model_and_tokenizer, EncoderWrapper, initialize_encoder_model_and_tokenizer_per_task
from sentence_transformers import SentenceTransformer
from dataset import load_qa_dataset_for_encoder_inference, load_nli_dataset_dl_for_inference
import gc
import torch
from eval import evaluate_sim, compute_query_matrix_and_norm, evaluate_f1_max, evaluate_exact_max, compute_acc_nli
from tqdm import tqdm
import pdb

def main(args):
    #initialize
    model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(args.model_name, args.task_type, args.is_tf, args.num_classification_labels)
    encoder_model = EncoderWrapper(model, tokenizer, args.task_type)
    

    if args.task_type == 'qa':
        queries_dl, same_lang_answers, cross_lang_answers, neg_same_lang_answers, neg_cross_lang_answers, hard_neg_same_lang_answers, hard_neg_cross_lang_answers, _ = load_qa_dataset_for_encoder_inference(args.dataset_file, args.batch_size)
        source_preds_per_layer = []
        target_preds_per_layer = []
        source_last_layer_preds = []
        target_last_layer_preds = []
        source_preds, target_preds = encoder_model.inference(queries_dl)
        
        # clear model to free up memory
        del encoder_model; del model
        gc.collect()
        torch.cuda.empty_cache()

        # evaluate
        if args.metrics == 'avg_cos_sim':
            sim_model = SentenceTransformer('sentence-transformers/LaBSE')
        else:
            sim_model = None

        if args.metrics == 'avg_cos_sim':
            source_query_embed, source_query_norm = compute_query_matrix_and_norm(sim_model, source_preds)
            target_query_embed, target_query_norm = compute_query_matrix_and_norm(sim_model, target_preds)

            same_lang_scores_avg = evaluate_sim(sim_model, source_query_embed, source_query_norm, same_lang_answers) # same language answers
            cross_lang_scores_avg = evaluate_sim(sim_model, target_query_embed, target_query_norm, cross_lang_answers) # cross language answers
            neg_same_lang_scores_avg = evaluate_sim(sim_model, source_query_embed, source_query_norm, neg_same_lang_answers) # same language answers
            neg_cross_lang_scores_avg = evaluate_sim(sim_model, target_query_embed, target_query_norm, neg_cross_lang_answers) # cross language answers
            hard_neg_same_lang_scores_avg = evaluate_sim(sim_model, source_query_embed, source_query_norm, hard_neg_same_lang_answers) # same language answers
            hard_neg_cross_lang_scores_avg = evaluate_sim(sim_model, target_query_embed, target_query_norm, hard_neg_cross_lang_answers) # cross language answers

        elif args.metrics == 'f1_max':
            same_lang_scores_avg = evaluate_f1_max(source_preds, same_lang_answers, args.source_lang)
            neg_same_lang_scores_avg = evaluate_f1_max(source_preds, neg_same_lang_answers, args.source_lang)
            hard_neg_same_lang_scores_avg = evaluate_f1_max(source_preds, hard_neg_same_lang_answers, args.source_lang)

            cross_lang_scores_avg = evaluate_f1_max(target_preds, cross_lang_answers, args.target_lang)
            neg_cross_lang_scores_avg = evaluate_f1_max(target_preds, neg_cross_lang_answers, args.target_lang)
            hard_neg_cross_lang_scores_avg = evaluate_f1_max(target_preds, hard_neg_cross_lang_answers, args.target_lang)
        
        else:
            same_lang_scores_avg = evaluate_exact_max(source_preds, same_lang_answers, args.source_lang)
            neg_same_lang_scores_avg = evaluate_exact_max(source_preds, neg_same_lang_answers, args.source_lang)
            hard_neg_same_lang_scores_avg = evaluate_exact_max(source_preds, hard_neg_same_lang_answers, args.source_lang)

            cross_lang_scores_avg = evaluate_exact_max(target_preds, cross_lang_answers, args.target_lang)
            neg_cross_lang_scores_avg = evaluate_exact_max(target_preds, neg_cross_lang_answers, args.target_lang)
            hard_neg_cross_lang_scores_avg = evaluate_exact_max(target_preds, hard_neg_cross_lang_answers, args.target_lang)
        
        print(f"Positive same language scores average: {same_lang_scores_avg}")
        print(f"Positive cross language scores average: {cross_lang_scores_avg}")
        print()
        print(f"Negative same language scores average: {neg_same_lang_scores_avg}")
        print(f"Negative cross language scores average: {neg_cross_lang_scores_avg}")
        print()
        print(f"Hard negative same language scores average: {hard_neg_same_lang_scores_avg}")
        print(f"Hard negative cross language scores average: {hard_neg_cross_lang_scores_avg}")
            



    
    # nli
    else:
        nli_dl, gts, _ = load_nli_dataset_dl_for_inference(args.dataset_file, args.batch_size)
        preds, _ = encoder_model.inference(nli_dl)
        accuracy = compute_acc_nli(preds, gts)
        print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_type', type=str, default='qa', choices=['nli', 'qa', 'nli_qa', 'mlama'])
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--is_tf', action='store_true')
    parser.add_argument('--metrics', type=str, default='f1_max', choices=['avg_cos_sim','f1_max', 'exact_max'])
    parser.add_argument('--num_classification_labels', type=int, default=3)


    args = parser.parse_args()

    main(args)