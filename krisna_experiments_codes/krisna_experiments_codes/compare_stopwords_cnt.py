from argparse import ArgumentParser
import pickle
from stop_words import get_stop_words
import collections
import pdb

stop_words_per_language = dict()
def compile_all_sentences_per_id(cs_data, mono_data, attribute_field):
    examples = dict()
    for row in cs_data:
        id = row['id']
        if id not in examples:
            examples[id] = {
                'mono_sentence':'',
                'cs_sentences':[]
            }
        examples[id]['cs_sentences'].append(row[attribute_field])
    for row in mono_data:
        id = row['id']
        if id in examples:
            examples[id]['mono_sentence'] = row[attribute_field]
    valid_examples = {id:val for id, val in examples.items() if val['mono_sentence']!=''}
    return valid_examples

def get_cs_stopwords_proba(cs_sentence_tokens, mono_sentence_tokens, lang):
    global stop_words_per_language
    cs_tokens_freq = dict(collections.Counter(cs_sentence_tokens))
    mono_tokens_freq = dict(collections.Counter(mono_sentence_tokens))
    possible_cs_og_tokens_freq = dict()
    for token, freq in mono_tokens_freq.items():
        if token not in cs_tokens_freq:
            possible_cs_og_tokens_freq[token] = freq
        else:
            possible_cs_og_tokens_freq[token] = mono_tokens_freq[token]-freq
    all_cs_tokens_len = 0
    stop_cs_tokens_len = 0
    for token, freq in possible_cs_og_tokens_freq.items():
        all_cs_tokens_len += freq
        if token in stop_words_per_language[lang]:
            stop_cs_tokens_len += freq
    if all_cs_tokens_len == 0:
        return 0
    return stop_cs_tokens_len/all_cs_tokens_len

     

def calculate_cs_stopwords(cs_data, mono_data, attribute_field, lang):
    all_examples = compile_all_sentences_per_id(cs_data, mono_data, attribute_field)
    all_cs_stopwords_proba = []
    for example in all_examples.values():
        mono_sentence_tokens = example['mono_sentence'].split()
        for cs_sentence in example['cs_sentences']:
            cs_sentence_tokens = cs_sentence.split()
            cs_stop_proba = get_cs_stopwords_proba(cs_sentence_tokens, mono_sentence_tokens, lang)
            all_cs_stopwords_proba.append(cs_stop_proba)
    return sum(all_cs_stopwords_proba)/len(all_cs_stopwords_proba)


        




def main(args):
    global stop_words_per_language
    stop_words_per_language = {
        'en': get_stop_words('en'),
        'de': get_stop_words('de'),
        'ar': get_stop_words('ar')
    }

    with open(args.mono_file1, 'rb') as f:
        mono_data1 = pickle.load(f)
    
    with open(args.mono_file2, 'rb') as f:
        mono_data2 = pickle.load(f) 
    
    with open(args.gcm_file, 'rb') as f:
        gcm_data = pickle.load(f)
    
    with open(args.rand_file, 'rb') as f:
        rand_data = pickle.load(f)

    rand_stopwords_proba = calculate_cs_stopwords(rand_data, mono_data1, args.attribute_field, 'ar')
    gcm_stopwords_proba = calculate_cs_stopwords(gcm_data, mono_data1, args.attribute_field, 'ar')
    gcm_stopwords_proba2 = calculate_cs_stopwords(gcm_data, mono_data2, args.attribute_field, 'en')
    print(f'Random CS stopwords conditional probability mean: {rand_stopwords_proba:.4f}')
    print(f'GCM CS stopwords conditional probability mean: {gcm_stopwords_proba:.4f} (ar), {gcm_stopwords_proba2:.4f} (en)')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--attribute_field', type=str, default='query')
    parser.add_argument('--mono_file1', type=str)
    parser.add_argument('--mono_file2', type=str)
    parser.add_argument('--gcm_file', type=str)
    parser.add_argument('--rand_file', type=str)


    args = parser.parse_args()

    main(args)
