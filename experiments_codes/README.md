# Crosslingual Consistency of Multilingual Models

This repository provides the official implementation of paper titled "Do Multilingual Language Models Show Crosslingual Knowledge Consistency?"

## Setup
1. Install depdendencies 
```
conda create -n crosslingual-knowledge-consistency
conda activate crosslingual-knowledge-consistency
pip install -r requirements.txt
```

2. Please copy all predictions and analyses assets from **. Then extract each compressed to file into one dedicated folder

## How to reproduce the results
1. To get the predictions on every encoder layer you can execute this following command (set the probed layers to just -1 if we just want to extract the results from the last layer of the encoder)
```
python measure_consistency_mlama.py --batch_size 8 --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name facebook/xlm-v-base --matrix_lang en --embedded_lang ta --output_prefix evaluations/mlama-xlm-v-consistency --beam_topk 5 --ranking_topk 5
```

2. To get the predictions on every encoder layer by doing our causal intervention, you can set the suppression_constant value (0-1.0) for using the cauasal internvetion on attention weight and/or intervened_ffn_layers for the mlp activation pacthing like example below (please check the detail on our paper for the actual hyperparameters)
```
python measure_consistency_mlama.py --batch_size 8 --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name facebook/xlm-v-base --matrix_lang en --embedded_lang ta --output_prefix evaluations/mlama-xlm-v-causal-consistency --beam_topk 5 --ranking_topk 5 --suppression_constant 0.7 --intervened_ffn_layers 2 5 6
```

3. To get the all object-subject attention weights for every head in encoder, we can run this command
```
python measure_average_attention.py --batch_size 8 --probed_layers 0 1 2 3 4 5 --model_name xlm-roberta-base --matrix_lang en --embedded_lang de --output_prefix analysis/factors/mlama-xlm-roberta-base-attention
```

4. To obtain all ig2 scores for each neurons on every mlp part of selected encoder layers, we can execute a command like this
```
python measure_encoder_cm_bias.py  --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name microsoft/xlm-align-base --matrix_lang en --embedded_lang ta --output_prefix ./analysis/mlama-xlm-align --model_type encoder
```


5. To do the visualization of layer-wise consistency we can execute a command like this
```
python analysis.scripts.visualize_layerwise_crosslingual_consistencies --prediction_files analysis/evaluations/mlama-xlm-r-base-en-ta.pkl analysis/evaluations/mlama-xlm-r-base-en-ar.pkl --label_names xlm-r-base_en-ta xlm-r-base_en-ar --model_name xlm-r-base --rankc_filepath figures/mlama-xlm-r-base-layerwise_rankc.png --rankc_filepath figures/mlama-xlm-r-base-layerwise_accuracy.png 
```

6. To do the visualization of overall consistency we can execute a command like this
```
python analysis.scripts.visualize_overall_crosslingual_consistencies --model_name xlm-roberta-base --modified_model_name facebook/xlm-v-base --figures_folder figures --layer_to_pick 11
```
To save the figure, you cam just click the "camera" icon on the opened figure in the browser.

7. To get the heatmap chart for attention analysis, we can execute a command like this following example 
```
python analysis.scripts.visualize_heatmap_attention --model xlm-roberta-base --attention_folder analysis/factors --lang_pair en-ta --lang_pair_title En --figures_folder figures
```

8. To get the chart for ig^2 analysis, we can execute a command like this following example 
```
python analysis.scripts.visualize_ffn_bias --model xlm-roberta-base --ig2_folder analysis/factors --lang_pair en-ta --lang_pair_title En --figures_folder figures
```

9. To obtain the spearman rho correlation between attention weights and consistency scores, we can execute command like this
```
python analysis.scripts.measure_correlation_attention_ratio --predictions_files analysis/mlama-xlm-r-base-consistency-en-ta.pkl analysis/mlama-xlm-r-base-consistency-en-ar.pkl --cm_attn_files analysis/factors/mlama-xlm-r-base-attention_en-ta-cmix.pkl analysis/factors/mlama-xlm-r-base-attention_en-ar-cmix.pkl --mono_attn_files analysis/factors/mlama-xlm-r-base-attention_en-ta-mono.pkl analysis/factors/mlama-xlm-r-base-attention_en-ar-mono.pkl   
```

10.  To obtain the spearman rho correlation between attention weights and IG2 scores, we can execute command like this
```
python analysis.scripts.measure_correlation_attention_ratio --predictions_files analysis/mlama-xlm-r-base-consistency-en-ta.pkl analysis/mlama-xlm-r-base-consistency-en-ar.pkl --cm_ig2_files analysis/factors/mlama-xlm-r-base-ig2_en-ta-cmix.pkl analysis/factors/mlama-xlm-r-base-ig2_en-ar-cmix.pkl --mono_ig2_files analysis/factors/mlama-xlm-r-base-ig2_en-ta-mono.pkl analysis/factors/mlama-xlm-r-base-ig2_en-ar-mono.pkl   
``` 