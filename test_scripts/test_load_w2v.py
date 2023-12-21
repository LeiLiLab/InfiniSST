import os
import fairseq
from fairseq.models.wav2vec import Wav2VecEncoder
from fairseq.models.hubert import HubertEncoder
# wav2vec 2.0
'''
state = fairseq.checkpoint_utils.load_checkpoint_to_cpu('/home/zhhao/ssl_model/20230627/wav2vec_vox_960h_pl.pt')
print(state['cfg']['model']['w2v_args']['model'].encoder_embed_dim)

encoder = Wav2VecEncoder(state['cfg']['model'], None)
new = {}

for key in state['model'].keys():
    new_key = key.replace('w2v_encoder.', '')
    if not new_key.startswith('proj'):
        new[new_key] = state['model'][key]
encoder.load_state_dict(new, strict=True)

length_after_ss = encoder.w2v_model._get_feat_extract_output_lengths
extract_features = encoder.w2v_model.extract_features
'''
# hubert
'''
#https://github.com/facebookresearch/fairseq/issues/4003
os.makedirs('/tmp/hubert_labels', exist_ok=True)
with open('/tmp/hubert_labels/dict.lyr9.km500.txt', 'w') as file:
    for i in range(500):
        file.write(f"{i} 1\n")
state = fairseq.checkpoint_utils.load_checkpoint_to_cpu('/home/zhhao/ssl_model/20230627/hubert_large_ll60k_finetune_ls960.pt')
state['cfg']['model']['w2v_args']['task']['label_dir'] = "/tmp/hubert_labels"

encoder = HubertEncoder(state['cfg']['model'], None)
new = {}

for key in state['model'].keys():
    new_key = key.replace('w2v_encoder.', '')
    if not new_key.startswith('proj'):
        new[new_key] = state['model'][key]
encoder.load_state_dict(new, strict=True)
'''
# data2vec
state = fairseq.checkpoint_utils.load_checkpoint_to_cpu('/home/xixu/ssl_model/20230627/large_vox_960h.pt')
print(state['cfg']['model'])
encoder = Wav2VecEncoder(state['cfg']['model'], None)

