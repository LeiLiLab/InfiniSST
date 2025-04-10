import torch

from model.llama31 import SpeechLlamaModel

def beam_search_pseudo(
    model, 
    tokenizer,
    input_ids, 
    speech_batch, 
    multiplier, 
    num_beams, 
    max_new_tokens, 
    states,
):
    bsz = input_ids.size(0)

    # encode speech features
    model.model.speech_encoder.set_blocksize(multiplier)
    if states is None:
        speech_features, _ = model.model.speech_encoder.encode_speech(speech_batch)
    else:
        speech_features, states.speech_cache = model.model.speech_encoder.encode_speech(
            speech_batch, 
            cache=states.speech_cache
        )

    # create input embeddings
    inputs_embeds = model.model.embed_tokens(input_ids)
    indices = torch.arange(input_ids.shape[1], device=input_ids.device)
    filled_inputs_embeds = []
    for i in range(input_ids.size(0)):
        user_mask = input_ids[i] == model.config.user_token_id
        user_pos = indices[user_mask]

        assist_mask = input_ids[i] == model.config.assist_token_id
        assist_pos = indices[assist_mask]

        user_pos = [
            pos for pos in user_pos if input_ids[i, pos - 1] == model.config.start_header_id
        ]
        assist_pos = [
            pos for pos in assist_pos if input_ids[i, pos - 1] == model.config.start_header_id
        ]

        filled_inputs_embed = inputs_embeds[i]
        index = 0
        for u_p, a_p in zip(user_pos, assist_pos):
            filled_inputs_embed = torch.cat(
                [
                    filled_inputs_embed[: u_p + 3],
                    speech_features[i, index : index + a_p - u_p - 5],
                    filled_inputs_embed[a_p - 2 :]
                ],
                dim=0                            
            )
            index += a_p - u_p - 5
        filled_inputs_embeds.append(filled_inputs_embed)
    inputs_embeds = torch.stack(filled_inputs_embeds)

    # prefill
    prefill_outputs = super(SpeechLlamaModel, model.model).forward(
        input_ids=None, 
        attention_mask=None,
        past_key_values=states.past_key_values,
        inputs_embeds=inputs_embeds, 
        use_cache=True,
        output_attentions=False, 
        output_hidden_states=False,
    )
    hidden_states = prefill_outputs.last_hidden_state
    past_key_values = list(prefill_outputs.past_key_values)
    logits = model.lm_head(hidden_states)[:, -1, :]
    logps = torch.log_softmax(logits, dim=-1)
    ## pick top-beam candidates
    topk_logps, topk_indices = torch.topk(logps, num_beams, dim=-1)
    ## replicate kv cache
    for i, (k, v) in enumerate(past_key_values):
        past_key_values[i] = (
            k.repeat_interleave(num_beams, dim=0),
            v.repeat_interleave(num_beams, dim=0)
        )
    
    # start beam search
    sum_logps = topk_logps.view(-1)
    num_remaining_beams = [num_beams] * bsz
    generated_ids = topk_indices.view(-1, 1)
    results = [[] for _ in range(bsz)]
    for _ in range(max_new_tokens):
        # collect finished beams
        idx = 0
        mask = torch.ones(sum(num_remaining_beams), dtype=torch.bool, device=generated_ids.device)
        for i in range(bsz):
            for j in range(num_remaining_beams[i]):
                if generated_ids[idx, -1] == tokenizer.eos_token_id:
                    num_remaining_beams[i] -= 1
                    mask[idx] = False
                    kv_cache = []
                    for k, v in past_key_values:
                        kv_cache.append((k[[idx]], v[[idx]]))
                    results[i].append({
                        "sequences": generated_ids[idx].tolist(),
                        "logp": sum_logps[idx] / len(generated_ids[idx]),
                        "past_key_values": kv_cache
                    })
                idx += 1
        
        generated_ids = generated_ids[mask]
        sum_logps = sum_logps[mask]
        for i, (k, v) in enumerate(past_key_values):
            past_key_values[i] = (k[mask], v[mask])

        if sum(num_remaining_beams) == 0:
            break

        decoder_input_ids = generated_ids[:, -1:]
        decoder_input_embs = model.model.embed_tokens(decoder_input_ids)
        decoder_outputs = super(SpeechLlamaModel, model.model).forward(
            input_ids=None,
            attention_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=decoder_input_embs,
            use_cache=True,
            output_attentions=False, 
            output_hidden_states=False,
        )
        hidden_states = decoder_outputs.last_hidden_state
        past_key_values = list(decoder_outputs.past_key_values)
        logits = model.lm_head(hidden_states)[:, -1, :]
        logps = torch.log_softmax(logits, dim=-1)

        idx = 0
        new_generated_ids = []
        new_sum_logps = []
        kv_cache_indices = []
        for i in range(bsz):
            logp = logps[idx : idx + num_remaining_beams[i]]
            topk_logp, topk_indices = torch.topk(logp, num_remaining_beams[i], dim=-1)
            topk_logp = topk_logp.view(-1)
            topk_indices = topk_indices.view(-1)

            sum_logp = sum_logps[idx : idx + num_remaining_beams[i]]
            sum_logp = sum_logp.repeat_interleave(num_remaining_beams[i])
            sum_logp += topk_logp

            topk_sum_logp, topk_sum_indices = sum_logp.topk(num_remaining_beams[i], dim=-1)
            new_sum_logps.append(topk_sum_logp)
            beam_idx = topk_sum_indices // num_remaining_beams[i]
            for j in range(num_remaining_beams[i]):
                prev_ids = generated_ids[idx + beam_idx[j]]
                new_id = topk_indices[topk_sum_indices[j]]
                new_generated_ids.append(prev_ids.tolist() + [new_id.item()])
                kv_cache_idx = idx + beam_idx[j].item()
                kv_cache_indices.append(kv_cache_idx)

            idx += num_remaining_beams[i]

        sum_logps = torch.cat(new_sum_logps, dim=0)
        generated_ids = torch.tensor(new_generated_ids).to(generated_ids)
        for i, (k, v) in enumerate(past_key_values):
            past_key_values[i] = (k[kv_cache_indices], v[kv_cache_indices])

    idx = 0
    for i in range(bsz):
        for j in range(num_remaining_beams[i]):
            kv_cache = []
            for k, v in past_key_values:
                kv_cache.append((k[[idx]], v[[idx]]))                    
            results[i].append({
                "sequences": generated_ids[idx].tolist(),
                "logp": sum_logps[idx] / len(generated_ids[idx]),
                "past_key_values": kv_cache
            })

    for i in range(bsz):
        results[i] = sorted(results[i], key=lambda x: x["logp"], reverse=True)[0]

    return results