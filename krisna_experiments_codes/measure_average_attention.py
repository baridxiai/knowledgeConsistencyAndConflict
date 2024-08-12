from argparse import ArgumentParser
import pickle
from utils import initialize_wrapped_model_and_tokenizer, load_mlama

def main(args):
    task_type = 'cloze'
    wrapped_model, _ = initialize_wrapped_model_and_tokenizer(args.model_name, task_type)
    mlama_instances = load_mlama(args.matrix_lang, args.embedded_lang)

    mono_attentions, cs_attentions = wrapped_model.extract_attention_scores_subj_obj(mlama_instances, args.batch_size, args.probed_layers)

    mono_filepath = f"{args.output_prefix}_matrix-{args.matrix_lang}-embedded-{args.embedded_lang}-mono-attentions.pkl"
    cs_filepath = f"{args.output_prefix}_matrix-{args.matrix_lang}-embedded-{args.embedded_lang}-cm-attentions.pkl"
    
    with open(mono_filepath, 'wb') as f:
        pickle.dump(mono_attentions, f)
    
    with open(cs_filepath, 'wb') as f:
        pickle.dump(cs_attentions, f)
         

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--probed_layers', type=int, nargs='+', default=[], help='Which layer(s) do you want to analyze')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--matrix_lang', type=str)
    parser.add_argument('--embedded_lang', type=str)
    parser.add_argument('--output_prefix', type=str)

    args = parser.parse_args()

    main(args)