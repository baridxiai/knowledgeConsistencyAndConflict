from argparse import ArgumentParser
import pickle
from tqdm import tqdm

def main(args):
    source_nli_data = []
    target_nli_data = []
    with open(args.qa_dataset_file, 'rb') as f:
        data_obj = pickle.load(f)

    for instance in tqdm(data_obj):
        # source nli
        source_id = instance[]
        source_premise = instance['query']
        if instance['context_same_lang'] != '':
            source_premise = f"{instance['context_same_lang']}. {source_premise}"
        pos_source_hypotheses = instance['answers_same_lang']
        neg_source_hypotheses = instance['negative_answers_same_lang']
        hard_neg_source_hypotheses = instance['hard_negative_answers_same_lang']
        for pos_source_hypothesis in pos_source_hypotheses:
            source_nli_data.append({
                'premise': source_premise,
                'hypothesis': pos_source_hypothesis,
                'label': 0
            })
        for neg_source_hypothesis in neg_source_hypotheses:
            source_nli_data.append({
                'premise': source_premise,
                'hypothesis': neg_source_hypothesis,
                'label': 1
            })
        for hard_neg_source_hypothesis in hard_neg_source_hypotheses:
            source_nli_data.append({
                'premise': source_premise,
                'hypothesis': hard_neg_source_hypothesis,
                'label': 1
            })

        # target nli
        target_premise = instance['query']
        if instance['context_cross_lang'] != '':
            target_premise = f"{instance['context_cross_lang']}. {target_premise}"
        pos_target_hypotheses = instance['answers_cross_lang']
        neg_target_hypotheses = instance['negative_answers_cross_lang']
        hard_neg_target_hypotheses = instance['hard_negative_answers_cross_lang']
        for pos_target_hypothesis in pos_target_hypotheses:
            target_nli_data.append({
                'premise': target_premise,
                'hypothesis': pos_target_hypothesis,
                'label': 0
            })
        for neg_target_hypothesis in neg_target_hypotheses:
            target_nli_data.append({
                'premise': target_premise,
                'hypothesis': neg_target_hypothesis,
                'label': 1
            })
        for hard_neg_target_hypothesis in hard_neg_target_hypotheses:
            target_nli_data.append({
                'premise': target_premise,
                'hypothesis': hard_neg_target_hypothesis,
                'label': 1
            })
    
    with open(args.output_source_lang_file, 'wb') as f:
        pickle.dump(source_nli_data, f)

    with open(args.output_target_lang_file, 'wb') as f:
        pickle.dump(target_nli_data, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--qa_dataset_file', type=str)
    parser.add_argument('--output_source_lang_file', type=str)
    parser.add_argument('--output_target_lang_file', type=str)


    args = parser.parse_args()

    main(args)