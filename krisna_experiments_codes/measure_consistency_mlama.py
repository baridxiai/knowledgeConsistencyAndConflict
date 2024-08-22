from argparse import ArgumentParser
from eval import compute_rankc
import pickle
from utils import initialize_wrapped_model_and_tokenizer, load_mlama 

def main(args):
    task_type = 'cloze'
    use_custom_bias_model = args.suppression_constant is not None or args.intervened_ffn_layers is not None
    wrapped_model, _ = initialize_wrapped_model_and_tokenizer(args.model_name, task_type, use_custom_bias_model)
    mlama_instances = load_mlama(args.matrix_lang, args.embedded_lang)

    if args.intervened_ffn_layers is not None or args.suppression_constant is not None:
        mono_rank_preds, cs_rank_preds, gts = wrapped_model.inference_cloze_task_with_causal_intervention(mlama_instances, args.batch_size, args.probed_layers, args.beam_topk, args.ranking_topk, args.intervened_ffn_layers, args.suppression_constant)
    else:
        mono_rank_preds, cs_rank_preds, gts = wrapped_model.inference_cloze_task(mlama_instances, args.batch_size, args.probed_layers, args.beam_topk, args.ranking_topk)

    if len(args.probed_layers) == 0 or -1 in args.probed_layers:
        print(f"Matrix Language: {args.matrix_lang}, Embedded Language: {args.embedded_lang}")
        print(f"RankC score: {compute_rankc(cs_rank_preds, mono_rank_preds)}")

    else:
        out_dict = dict()
        for layer in args.probed_layers:
            rankc_score = compute_rankc(cs_rank_preds[layer], mono_rank_preds[layer])
            mono_mrr = compute_mrr(mono_rank_preds[layer], gts)
            cs_mrr = compute_mrr(cs_rank_preds[layer], gts)
            out_dict[layer] = {
                'rankC': rankc_score,
                'mono_rank_preds': mono_rank_preds[layer],
                'cs_rank_preds': cs_rank_preds[layer]
            }

            print(f"Layer: {layer}")
            print(f"Matrix Language: {args.matrix_lang}, Embedded Language: {args.embedded_lang}")
            print(f"RankC score: {rankc_score}")

        if args.suppression_constant is not None and args.intervened_ffn_layers is not None and len(args.intervened_ffn_layers) > 0:
            out_filepath = f"{args.output_prefix}_matrix-{args.matrix_lang}-embedded-{args.embedded_lang}-both-intervention.pkl"
        elif args.suppression_constant is not None:
            out_filepath = f"{args.output_prefix}_matrix-{args.matrix_lang}-embedded-{args.embedded_lang}-attn-intervention.pkl"
        elif args.intervened_ffn_layers is not None:
            out_filepath = f"{args.output_prefix}_matrix-{args.matrix_lang}-embedded-{args.embedded_lang}-ffn-intervention.pkl"
        
        with open(out_filepath, 'wb') as f:
            pickle.dump(out_dict, f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--probed_layers', type=int, nargs='+', default=[])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--matrix_lang', type=str)
    parser.add_argument('--embedded_lang', type=str)
    parser.add_argument('--output_prefix', type=str, required=False)
    parser.add_argument('--beam_topk', type=int, default=1)
    parser.add_argument('--ranking_topk', type=int, default=3)
    parser.add_argument('--suppression_constant', type=float, default=None)
    parser.add_argument('--intervened_ffn_layers', type=int, nargs='+', default=None)


    args = parser.parse_args()

    main(args)
