from typing import List

import torch

from model.flashinfer.engine import (
    pop_paged_kv_cache,
    copy_paged_kv_cache,
    move_paged_kv_cache,
    duplicate_paged_kv_cache,
    SpeechCache,
    LLMCache
)

from model.qwen25 import SpeechQwenModel

class BeamState:
    def __init__(self, num_beams):
        self.num_beams = num_beams
        self.num_remaining_beams = num_beams
        self.sum_logps = None
        self.generated_ids = None
        self.results = []

class Request:
    def __init__(
        self, 
        input_ids, 
        speech, 
        blocksize, 
        max_new_tokens, 
        speech_max_steps, 
        speech_cache, 
        llm_max_steps, 
        llm_max_steps_start, 
        llm_cache
    ):
        self.input_ids = input_ids
        self.speech = speech
        self.blocksize = blocksize
        self.max_new_tokens = max_new_tokens
        
        self.speech_max_steps = speech_max_steps
        self.speech_cache = speech_cache

        self.llm_max_steps = llm_max_steps
        self.llm_max_steps_start = llm_max_steps_start
        self.llm_cache = llm_cache

        self.prefill_finished = False
        self.decode_finished = False
        self.beam_state = None

    def get_speech_request(self):
        return {
            "speech": self.speech,
            "blocksize": self.blocksize,
            "cache": self.speech_cache
        }
    
    def get_llm_request(self):
        return {
            "input_ids": self.input_ids,
            "cache": self.llm_cache,
        }

def collect_finished_beams(request, tokenizer, length_penalty):
    remaining_llm_cache = []
    mask = [True] * request.beam_state.num_remaining_beams
    for j in range(request.beam_state.num_remaining_beams):
        gen_j = request.beam_state.generated_ids[j]
        if gen_j[-1] == tokenizer.eos_token_id or len(gen_j) >= request.max_new_tokens:
            request.beam_state.num_remaining_beams -= 1
            mask[j] = False
            request.beam_state.results.append({
                "sequence": gen_j.tolist(),
                "logp": request.beam_state.sum_logps[j] / (len(gen_j) ** length_penalty),
                "cache": request.llm_cache[j]
            })
        else:
            remaining_llm_cache.append(request.llm_cache[j])
    request.llm_cache = remaining_llm_cache
    request.beam_state.sum_logps = request.beam_state.sum_logps[mask]
    request.beam_state.generated_ids = request.beam_state.generated_ids[mask]


def finish_beam_search(request, llm_decode_pagetable, llm_prefill_pagetable):
    assert request.beam_state.num_remaining_beams == 0
    results = sorted(request.beam_state.results, key=lambda x: x["logp"], reverse=True)
    for r in results[1:]:
        llm_decode_pagetable, _, _ = pop_paged_kv_cache(
            llm_decode_pagetable,
            r['cache'].paged_kv_indices,
            r['cache'].paged_kv_last_page_len,
            0,
        )
    request.results = results[0]
    request.llm_cache = results[0]['cache']
    # trim llm kv cache
    llm_decode_pagetable, request.llm_cache.paged_kv_indices, request.llm_cache.paged_kv_last_page_len = \
        pop_paged_kv_cache(
            llm_decode_pagetable,
            request.llm_cache.paged_kv_indices,
            request.llm_cache.paged_kv_last_page_len,
            request.llm_max_steps,
            request.llm_max_steps_start,
        )
    # move llm kv cache to prefill
    llm_decode_pagetable, llm_prefill_pagetable, request.llm_cache.paged_kv_indices, request.llm_cache.paged_kv_last_page_len = \
        move_paged_kv_cache(
            request.llm_cache.paged_kv_indices,
            request.llm_cache.paged_kv_last_page_len,
            llm_decode_pagetable,
            llm_prefill_pagetable
        )
    request.decode_finished = True


def prefill(
    requests,
    model,
    tokenizer,
    num_beams,
    length_penalty,
    speech_pagetable,
    llm_prefill_pagetable,
    llm_decode_pagetable,
):
    bsz = len(requests)
    speech_requests = [request.get_speech_request() for request in requests]
    speech_features, speech_requests, speech_pagetable, _ = model.model.speech_encoder.encode_speech_fast(
        speech_requests,
        speech_pagetable,
    )

    llm_requests = [request.get_llm_request() for request in requests] 

    logits, llm_requests, llm_prefill_pagetable, layer_results = model(
        llm_requests,
        llm_prefill_pagetable,
        speech_features,
    )
    logps = torch.log_softmax(logits, dim=-1)
    topk_logps, topk_indices = torch.topk(logps, num_beams, dim=-1)

    decode_device = llm_decode_pagetable.paged_kv_cache.device
    for i, request in enumerate(requests):
        request.speech_cache = speech_requests[i]['cache']
        request.llm_cache = llm_requests[i]['cache']

        # initialize beam search state
        request.beam_state = BeamState(num_beams)
        request.beam_state.sum_logps = topk_logps[i].view(-1).to(decode_device)
        request.beam_state.generated_ids = topk_indices[i].view(-1, 1).to(decode_device)
        request.beam_state.results = []

        # trim speech kv cache
        speech_cache = request.speech_cache
        speech_pagetable, speech_cache.paged_kv_indices, speech_cache.paged_kv_last_page_len = \
            pop_paged_kv_cache(
                speech_pagetable,
                speech_cache.paged_kv_indices,
                speech_cache.paged_kv_last_page_len,
                request.speech_max_steps,
            )
        
        # move llm kv cache to decode
        llm_cache = request.llm_cache
        llm_prefill_pagetable, llm_decode_pagetable, llm_cache.paged_kv_indices, llm_cache.paged_kv_last_page_len = \
            move_paged_kv_cache(
                llm_cache.paged_kv_indices,
                llm_cache.paged_kv_last_page_len,
                llm_prefill_pagetable,
                llm_decode_pagetable
            )
        
        # TODO: share prefix kv cache
        # replicate kv cache for each beam
        beam_cache = [llm_cache]
        for i in range(1, num_beams):
            cache_i = LLMCache()            
            llm_decode_pagetable, cache_i.paged_kv_indices, cache_i.paged_kv_last_page_len = \
                duplicate_paged_kv_cache(
                    llm_cache.paged_kv_indices,
                    llm_cache.paged_kv_last_page_len,
                    llm_decode_pagetable,
                )
            beam_cache.append(cache_i)
        request.llm_cache = beam_cache

        # finish prefill
        request.prefill_finished = True

    return requests, speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable

def decode(
    requests, 
    model, 
    tokenizer, 
    num_beams, 
    length_penalty,
    speech_pagetable,
    llm_prefill_pagetable,
    llm_decode_pagetable
):       
    # collect finished beams
    mask = []
    finished_requests = []
    remaining_requests = []
    for i in range(len(requests)):
        collect_finished_beams(requests[i], tokenizer, length_penalty)
        if requests[i].beam_state.num_remaining_beams > 0:
            remaining_requests.append(requests[i])
            mask.append(True)
        else:
            finish_beam_search(
                requests[i], 
                llm_decode_pagetable, 
                llm_prefill_pagetable
            )
            finished_requests.append(requests[i])
            mask.append(False)
        
    if sum(mask) == 0:
        return requests, speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable

    bsz = len(remaining_requests)
    sum_logps = torch.cat([request.beam_state.sum_logps for request in remaining_requests], dim=0)
    num_remaining_beams = [request.beam_state.num_remaining_beams for request in remaining_requests]

    llm_requests = []
    for request in remaining_requests:
        for j in range(request.beam_state.num_remaining_beams):
            llm_requests.append({
                "input_ids": request.beam_state.generated_ids[j][-1:],
                "cache": request.llm_cache[j]
            })

    logits, llm_requests, llm_decode_pagetable, _ = model(
        llm_requests,
        llm_decode_pagetable,
    )
    logps = torch.log_softmax(logits, dim=-1)

    idx = 0
    for i in range(bsz):
        logp = logps[idx : idx + num_remaining_beams[i]]
        topk_logp, topk_indices = torch.topk(logp, num_remaining_beams[i], dim=-1)
        topk_logp = topk_logp.view(-1)
        topk_indices = topk_indices.view(-1)

        sum_logp = sum_logps[idx : idx + num_remaining_beams[i]]
        sum_logp = sum_logp.repeat_interleave(num_remaining_beams[i])
        sum_logp += topk_logp

        topk_sum_logp, topk_sum_indices = sum_logp.topk(num_remaining_beams[i], dim=-1)
        remaining_requests[i].beam_state.sum_logps = topk_sum_logp

        new_generated_ids = []
        new_llm_cache = []
        beam_idx = topk_sum_indices // num_remaining_beams[i]
        for j in range(num_remaining_beams[i]):
            prev_ids = remaining_requests[i].beam_state.generated_ids[beam_idx[j]]
            new_id = topk_indices[topk_sum_indices[j]]
            new_generated_ids.append(prev_ids.tolist() + [new_id.item()])

            # TODO: share prefix kv cache
            cache_j = LLMCache()
            llm_decode_pagetable, cache_j.paged_kv_indices, cache_j.paged_kv_last_page_len = \
                duplicate_paged_kv_cache(
                    remaining_requests[i].llm_cache[beam_idx[j]].paged_kv_indices,
                    remaining_requests[i].llm_cache[beam_idx[j]].paged_kv_last_page_len,
                    llm_decode_pagetable,
                )
            new_llm_cache.append(cache_j)

        # pop previous beam kv cache
        for j in range(num_remaining_beams[i]):
            llm_decode_pagetable, _, _ = pop_paged_kv_cache(
                llm_decode_pagetable,
                remaining_requests[i].llm_cache[j].paged_kv_indices,
                remaining_requests[i].llm_cache[j].paged_kv_last_page_len,
                0,
            )
        remaining_requests[i].llm_cache = new_llm_cache
            
        remaining_requests[i].beam_state.generated_ids = \
            torch.tensor(new_generated_ids).to(remaining_requests[i].beam_state.generated_ids)
        idx += num_remaining_beams[i]
    
    for i in range(len(remaining_requests)):
        collect_finished_beams(remaining_requests[i], tokenizer, length_penalty)
        if remaining_requests[i].beam_state.num_remaining_beams == 0:
            finish_beam_search(
                remaining_requests[i], 
                llm_decode_pagetable, 
                llm_prefill_pagetable
            )
    
    requests = []
    for m in mask:
        if m: requests.append(remaining_requests.pop(0))
        else: requests.append(finished_requests.pop(0))

    return requests, speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable

def beam_search(
    requests: List[Request],
    model,
    tokenizer,
    num_beams,
    length_penalty,
    speech_pagetable,
    llm_prefill_pagetable,
    llm_decode_pagetable,
):
    prefill_finished = requests[0].prefill_finished
    assert all(request.prefill_finished == prefill_finished for request in requests)

    if not prefill_finished:
        return prefill(
            requests, 
            model, 
            tokenizer, 
            num_beams, 
            length_penalty,
            speech_pagetable, 
            llm_prefill_pagetable,
            llm_decode_pagetable
        )
    else:
        return decode(
            requests, 
            model, 
            tokenizer, 
            num_beams, 
            length_penalty,
            speech_pagetable,
            llm_prefill_pagetable,
            llm_decode_pagetable
        )

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
                    filled_inputs_embed[: u_p + 2],
                    speech_features[i, index : index + a_p - u_p - 5],
                    filled_inputs_embed[a_p - 3 :]
                ],
                dim=0                            
            )
            index += a_p - u_p - 5
        filled_inputs_embeds.append(filled_inputs_embed)
    inputs_embeds = torch.stack(filled_inputs_embeds)

    # prefill
    prefill_outputs = super(SpeechQwenModel, model.model).forward(
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
        decoder_outputs = super(SpeechQwenModel, model.model).forward(
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
            idx += 1
    
    for i in range(bsz):
        results[i] = sorted(results[i], key=lambda x: x["logp"], reverse=True)[0]

    return results