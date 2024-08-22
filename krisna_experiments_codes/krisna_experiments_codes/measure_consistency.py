from argparse import ArgumentParser
from model import initialize_model_and_tokenizer, DecoderLensWrapper, EncoderWrapper, initialize_encoder_model_and_tokenizer_per_task
from sentence_transformers import SentenceTransformer
from dataset import load_qa_dataset_for_inference, load_nli_dataset_for_inference, load_qa_dataset_for_encoder_inference,  load_nli_dataset_dl_for_inference
import gc
import torch
from eval import evaluate_sim, compute_query_matrix_and_norm, evaluate_f1_mean, evaluate_f1_means_max, evaluate_accuracy, compute_query_matrix_and_norm_multi_preds, evaluate_sim_multi_preds
from tqdm import tqdm
import pdb
import pickle

def main(args):

    nli_labels = ['True', 'Neither', 'False'] # same as xnli True: entailment, Neither: Neutral, False: contradiction
    nli_classification_labels = 3 # same as xnli

    # last layer
    if -1 in args.probed_layers:
        refs = []
        preds = []
        gts = []
        pivot_ids, main_ids = [], []
        main_preds, pivot_pres = [], []
        # encoder-decoder
        if 'mt0' in args.model_name or 'mt5' in args.model_name:
            model, tokenizer = initialize_model_and_tokenizer(args.model_name)
            decoder_lens_model = DecoderLensWrapper(model, tokenizer)

            if args.task_type == 'qa':
                main_query_batches, _, main_gts, _, _, _, _, main_ids = load_qa_dataset_for_inference(args.checked_dataset_file, args.batch_size)
                pivot_query_batches, pivot_gts, _, _, _, _, _, pivot_ids = load_qa_dataset_for_inference(args.pivot_dataset_file, args.batch_size)
                _, main_preds = decoder_lens_model.decode(main_query_batches)
                pivot_preds, _ = decoder_lens_model.decode(pivot_query_batches)
                

            
            else:
                main_nli_pairs, main_gts, main_ids = load_nli_dataset_for_inference(args.checked_dataset_file)
                pivot_nli_pairs, pivot_gts, pivot_ids = load_nli_dataset_for_inference(args.pivot_dataset_file)
                main_preds, _ = decoder_lens_model.classify(main_nli_pairs, nli_labels, False)
                pivot_preds, _ = decoder_lens_model.classify(pivot_nli_pairs, nli_labels, False)

        
        # encoder
        else:
            if args.task_type == 'qa':
                model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(args.model_name, args.task_type)
            else:
                model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(args.model_name, args.task_type, False, nli_classification_labels)
            encoder_model = EncoderWrapper(model, tokenizer, args.task_type)
            
            if args.task_type == 'qa':
                main_query_batches, _, main_gts, _, _, _, _, main_ids = load_qa_dataset_for_encoder_inference(args.checked_dataset_file, args.batch_size)
                pivot_query_batches, pivot_gts, _, _, _, _, _, pivot_ids = load_qa_dataset_for_encoder_inference(args.pivot_dataset_file, args.batch_size)
                _, main_preds = encoder_model.inference(main_query_batches)
                pivot_preds, _ = encoder_model.inference(pivot_query_batches)

            else:
                main_nli_pairs_batches, main_gts, main_ids = load_nli_dataset_dl_for_inference(args.checked_dataset_file, args.batch_size)
                pivot_nli_pairs_batches, pivot_gts, pivot_ids = load_nli_dataset_dl_for_inference(args.pivot_dataset_file, args.batch_size)
                main_preds, _ = encoder_model.inference(main_nli_pairs_batches)
                pivot_preds, _ = encoder_model.inference(pivot_nli_pairs_batches)

        assert len(main_ids) == len(main_preds)
        assert len(pivot_ids) == len(pivot_preds)
        
        pred_per_id = dict()
        ref_per_id = dict()
        gt_per_id = dict()


        for instance_id, pred, gt in zip(main_ids, main_preds, main_gts):
            if instance_id not in pred_per_id:
                pred_per_id[instance_id] = []
            pred_per_id[instance_id].append(pred)
            gt_per_id[instance_id] = gt
        
        for instance_id, pred in zip(pivot_ids, pivot_preds):
            ref_per_id[instance_id] = pred
        

        for instance_id in pivot_ids:
            refs.append(ref_per_id[instance_id])
            preds.append(pred_per_id[instance_id])
            gts.append(gt_per_id[instance_id])
        
        
        if args.metrics == 'avg_cos_sim':
            # clear model to free up memory
            del model;del decoder_lens_model
            gc.collect()
            torch.cuda.empty_cache()


            sim_model = SentenceTransformer('sentence-transformers/LaBSE')

            ref_embed, ref_norm = compute_query_matrix_and_norm(sim_model, refs)
            consistency_scores = evaluate_sim(sim_model, ref_embed, ref_norm, preds)

            pred_embed, pred_norm = compute_query_matrix_and_norm_multi_preds(sim_model, preds)
            evaluation_scores = evaluate_sim_multi_preds(sim_model, pred_embed, pred_norm, gts)
        
        elif args.metrics == 'avg_acc':
            consistency_scores = evaluate_accuracy(refs, preds)
            evaluation_scores = evaluate_accuracy(gts, preds)
            
        else:

            consistency_scores = evaluate_f1_mean(refs, preds, args.pivot_lang)
            #pdb.set_trace()
            evaluation_scores = evaluate_f1_means_max(preds, gts, args.pivot_lang)

        print(f"Consistency scores: {consistency_scores}")
        print(f"Evaluation scores: {evaluation_scores}")

    else:
        all_layer_preds = dict()

        # encoder-decoder
        if 'mt0' in args.model_name or 'mt5' in args.model_name:
            model, tokenizer = initialize_model_and_tokenizer(args.model_name)
            decoder_lens_model = DecoderLensWrapper(model, tokenizer)
            
            if args.task_type == 'qa':
                main_query_batches, _, main_gts, _, _, _, _, main_ids = load_qa_dataset_for_inference(args.checked_dataset_file, args.batch_size)
                pivot_query_batches, pivot_gts, _, _, _, _, _, pivot_ids = load_qa_dataset_for_inference(args.pivot_dataset_file, args.batch_size)
                _, main_layerwise_preds = decoder_lens_model.decode_on_particular_hidden_states(main_query_batches, args.probed_layers)
                pivot_layerwise_preds, _ = decoder_lens_model.decode_on_particular_hidden_states(pivot_query_batches, args.probed_layers)
      
            
            # NLI
            else:
                main_nli_pairs, main_gts, main_ids = load_nli_dataset_for_inference(args.checked_dataset_file)
                pivot_nli_pairs, pivot_gts, pivot_ids = load_nli_dataset_for_inference(args.pivot_dataset_file)       
                main_layerwise_preds = decoder_lens_model.classify_on_particular_hidden_states(main_nli_pairs, nli_labels, False, args.probed_layers)
                pivot_layerwise_preds = decoder_lens_model.classify_on_particular_hidden_states(pivot_nli_pairs, nli_labels, False, args.probed_layers)
            

            
        # encoder
        else:
            if args.task_type == 'qa':
                model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(args.model_name, args.task_type)
            else:
                model, tokenizer = initialize_encoder_model_and_tokenizer_per_task(args.model_name, args.task_type, False, nli_classification_labels)
            encoder_model = EncoderWrapper(model, tokenizer, args.task_type)

            if args.task_type == 'qa':
                main_query_batches, _, main_gts, _, _, _, _, main_ids = load_qa_dataset_for_encoder_inference(args.checked_dataset_file, args.batch_size)
                pivot_query_batches, pivot_gts, _, _, _, _, _, pivot_ids = load_qa_dataset_for_encoder_inference(args.pivot_dataset_file, args.batch_size)
                _, main_layerwise_preds = encoder_model.inference_per_layer(main_query_batches, args.probed_layers)
                pivot_layerwise_preds, _ = encoder_model.inference_per_layer(pivot_query_batches, args.probed_layers)

            else:
                main_nli_pairs_batches, main_gts, main_ids = load_nli_dataset_dl_for_inference(args.checked_dataset_file, args.batch_size)
                pivot_nli_pairs_batches, pivot_gts, pivot_ids = load_nli_dataset_dl_for_inference(args.pivot_dataset_file, args.batch_size)
                main_layerwise_preds, _ = encoder_model.inference_per_layer(main_nli_pairs_batches, args.probed_layers)
                pivot_layerwise_preds, _ = encoder_model.inference_per_layer(pivot_nli_pairs_batches, args.probed_layers)

        for layer in args.probed_layers:
            main_preds = main_layerwise_preds[layer]
            pivot_preds = pivot_layerwise_preds[layer]
            all_layer_preds[layer] = (main_preds, pivot_preds)
            
        sim_model = None
        if args.metrics == 'avg_cos_sim':
            # clear model to free up memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

            sim_model = SentenceTransformer('sentence-transformers/LaBSE')

        output_dict_per_layer = dict()
        for key, pred_pair in all_layer_preds.items():
            refs = []
            preds = []
            gts = []
            print(f"Layer: {key}")
            

            main_preds, pivot_preds = pred_pair[0], pred_pair[1]

            assert len(main_ids) == len(main_preds)
            assert len(pivot_ids) == len(pivot_preds)

            pred_per_id = dict()
            ref_per_id = dict()
            gt_per_id = dict()


            for instance_id, pred, gt in zip(main_ids, main_preds, main_gts):
                if instance_id not in pred_per_id:
                    pred_per_id[instance_id] = []
                pred_per_id[instance_id].append(pred)
                gt_per_id[instance_id] = gt
            
            for instance_id, pred in zip(pivot_ids, pivot_preds):
                ref_per_id[instance_id] = pred
            

            for instance_id in pivot_ids:
                refs.append(ref_per_id[instance_id])
                preds.append(pred_per_id[instance_id])
                gts.append(gt_per_id[instance_id])
            
            consistency_scores = -1
            if args.metrics == 'avg_acc':
                consistency_scores = evaluate_accuracy(refs, preds)
                evaluation_scores = evaluate_accuracy(gts, preds)
            elif args.metrics == 'avg_cos_sim':
                ref_embed, ref_norm = compute_query_matrix_and_norm(sim_model, refs)
                consistency_scores = evaluate_sim(sim_model, ref_embed, ref_norm, preds)

                pred_embed, pred_norm = compute_query_matrix_and_norm_multi_preds(sim_model, preds)
                evaluation_scores = evaluate_sim_multi_preds(sim_model, pred_embed, pred_norm, gts)

            else:
                consistency_scores = evaluate_f1_mean(refs, preds, args.pivot_lang)
                evaluation_scores = evaluate_f1_means_max(preds, gts, args.pivot_lang)
            
            output_dict_per_layer[key] = {
                'consistency': consistency_scores,
                'correctness': evaluation_scores
            }

            print(f"Consistency scores: {consistency_scores:.4f}")
            print(f"Evaluation scores: {evaluation_scores:.2f}\n")

        with open(args.output_file, 'wb') as f:
            pickle.dump(output_dict_per_layer, f)






if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task_type', type=str, default='qa', choices=['qa', 'nli'])
    parser.add_argument('--checked_dataset_file', type=str)
    parser.add_argument('--pivot_dataset_file', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--pivot_lang', type=str, default='en')
    parser.add_argument('--probed_layers', type=int, nargs='+')
    parser.add_argument('--metrics', type=str, default='avg_cos_sim', choices=['avg_cos_sim','avg_f1', 'avg_acc'])
    parser.add_argument('--output_file', type=str, required=False)


    args = parser.parse_args()

    main(args)
