#!/bin/bash
# Compositional Generalization Experiments
# Baseline
# Training
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=66
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_4 --training_batch_size=200 --max_training_iterations=200000 --seed=49
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_5 --training_batch_size=200 --max_training_iterations=200000 --seed=50

# Exactly the same for k=5,10,50, but with the --k set to the respective amount AND the following seeds:
# k=5, seeds run 1: 66, run 2: 67, run 3: 82
# k=10, seeds run 1: 84, run 2: 94, run 3: 104
# k=50, seeds run 1: 104, run 2: 105, run 3: 106

# Testing
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_3 --resume_from_file=adverb_k_1_run_3/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=adverb_k_1_run_3.json --max_decoding_steps=120 &> adverb_k_1_run_3/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=resnet_output --resume_from_file=resnet_output/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=resnet_output.json --max_decoding_steps=120 --encoder=bilstm &> resnet_output/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=transformer_output --resume_from_file=transformer_output/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=transformer_output.json --max_decoding_steps=120 &> transformer_output/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=grid_resnet_transformer_out --resume_from_file=grid_resnet_transformer_out/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=grid_grounding.json --encoder=transformer --max_decoding_steps=120 --max_testing_examples=2000 --max_training_examples=10000 &> grid_resnet_transformer_out/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=transformer_convnet_out --resume_from_file=transformer_convnet_out/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=transformer_convnet.json --encoder=transformer --max_decoding_steps=120 &> transformer_convnet_out/test.txt
# GECA
# Exactly the same commands as for the baseline except change the --data_directory to data/GECA, the output_directory, and the seed.
# Seeds run 1: 77, run 2: 81, run 3: 83

# Target Lengths Experiments
# Training
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=106 --cnn_kernel_size=13 &> target_lengths_run_1/target_lengths_run.txt
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=116 --cnn_kernel_size=13 &> target_lengths_run_2/target_lengths_run.txt
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=126 --cnn_kernel_size=13&> target_lengths_run_3/target_lengths_run.txt

# Testing
python3.7 -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_1 --resume_from_file=target_lengths_run_1/model_best.pth.tar --splits=test,dev,target_lengths --output_file_name=target_lengths_predict_run_1.json --max_decoding_steps=120 --cnn_kernel_size=13 &> target_lengths_run_1/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_2 --resume_from_file=target_lengths_run_2/model_best.pth.tar --splits=test,dev,target_lengths --output_file_name=target_lengths_predict_run_2.json --max_decoding_steps=120 --cnn_kernel_size=13 &> target_lengths_run_2/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=test,dev,target_lengths --output_file_name=target_lengths_predict_run_3.json --max_decoding_steps=120 --cnn_kernel_size=13 &> target_lengths_run_3/test.txt