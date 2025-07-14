import argparse

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_fea_length', type=int, default=10000,
                        help='the maximum sequence length of the patch embeddings.')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=1, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples for a batch.')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs.')
    # parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')

    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=768, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=4, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=6, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=6, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.5, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=3, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=1, help='whether decoding constraint.')
    parser.add_argument('--suppress_UNK', type=int, default=1, help='suppress UNK tokens in the decoding.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
    parser.add_argument('--length_penalty', type=str, default='wu_0.9', help='length penality')
    parser.add_argument('--diversity_lambda', type=float, default=1.5, help='diversity lambda')

    parser.add_argument('--d1', type=int, default=768, help='devices.')
    parser.add_argument('--d2', type=int, default=768, help='devices.')
    parser.add_argument('--embeddings_path', type=str, default='/mnt/saarthak/datasets/REG_processed/20x_512px_0px_overlap/slide_features_titan', help='emb 1 path.')
    parser.add_argument('--embeddings_path_2', type=str, default='/mnt/saarthak/datasets/REG_processed/20x_512px_0px_overlap/features_conch_v15', help='emb 2 path.')
    parser.add_argument('--reports_json_path', type=str, default='/mnt/surya/train.json', help='reports path.')
    parser.add_argument('--ckpt_path', type=str, default='/mnt/surya/projects/Wsi-rgen/checkpoints/2', help='reports path.')
    parser.add_argument('--model_save_path', type=str, default='/mnt/surya/projects/Wsi-rgen/saved_models/mi_gen_titan_plus_conch1.5_20250714_1_model.ckpt',
                        help='model save path')
    parser.add_argument('--devices', type=str, default='1,2,3,4,5', help='devices.')
    parser.add_argument('--fastdevrun', type=bool, default='false', help='fast dev run.')
    args = parser.parse_args()
    return args