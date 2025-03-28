for lang in vi zh; do
    ~/miniconda3/bin/python measure_consistency_mlama.py --batch_size 64 --probed_layers 47 --model_name facebook/xlm-roberta-xxl  --matrix_lang en --embedded_lang $lang --output_prefix evaluations/xlm-r-xxl/$lang --beam_topk 5 --ranking_topk 5
done
