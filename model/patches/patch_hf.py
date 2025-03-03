import torch
import torch.nn as nn
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Callable, Optional, Tuple, Union

from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.generation.beam_search import BeamSearchScorer, BeamScorer, BeamHypotheses

from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationMixin,
    GenerateOutput,
    GenerateBeamOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput,
    _split_model_inputs,
    stack_model_outputs,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.utils import (
    is_torchdynamo_compiling,
    logging,
)
import torch.distributed as dist
import inspect
import warnings

logger = logging.get_logger(__name__)

def beam_search_process(
    self,
    input_ids: torch.LongTensor,
    next_scores: torch.FloatTensor,
    next_tokens: torch.LongTensor,
    next_indices: torch.LongTensor,
    pad_token_id: Optional[Union[int, torch.Tensor]] = None,
    eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
    beam_indices: Optional[torch.LongTensor] = None,
    group_index: Optional[int] = 0,
    decoder_prompt_len: Optional[int] = 0,
    past_key_values: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # add up to the length which the next_scores is calculated on (including decoder prompt)
    cur_len = input_ids.shape[-1] + 1
    batch_size = len(self._beam_hyps) // self.num_beam_groups

    if not (batch_size == (input_ids.shape[0] // self.group_size)):
        if self.num_beam_groups > 1:
            raise ValueError(
                f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                f"size of {self.group_size} is expected by the beam scorer."
            )
        else:
            raise ValueError(
                f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                f"{self.group_size} is expected by the beam scorer."
            )

    device = input_ids.device
    next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
    next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
    next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

    if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id = torch.tensor(eos_token_id)

    for batch_idx in range(batch_size):
        batch_group_idx = batch_idx * self.num_beam_groups + group_index
        if self._done[batch_group_idx]:
            if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
            if eos_token_id is None or pad_token_id is None:
                raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
            # pad the batch
            next_beam_scores[batch_idx, :] = 0
            next_beam_tokens[batch_idx, :] = pad_token_id
            next_beam_indices[batch_idx, :] = 0
            continue

        # next tokens for this sentence
        beam_idx = 0
        for beam_token_rank, (next_token, next_score, next_index) in enumerate(
            zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
        ):
            batch_beam_idx = batch_idx * self.group_size + next_index
            # add to generated hypotheses if end of sentence
            if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                # if beam_token does not belong to top num_beams tokens, it should not be added
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                if is_beam_token_worse_than_top_num_beams:
                    continue
                if beam_indices is not None:
                    beam_index = beam_indices[batch_beam_idx]
                    beam_index = beam_index + (batch_beam_idx,)
                else:
                    beam_index = None

                if past_key_values is not None:
                    cur_past_key_values = DynamicCache(len(past_key_values))
                    for i, (k, v) in enumerate(past_key_values):
                        cur_past_key_values.update(
                            k[batch_beam_idx : batch_beam_idx + 1], 
                            v[batch_beam_idx : batch_beam_idx + 1], 
                            i
                        )

                self._beam_hyps[batch_group_idx].add(
                    input_ids[batch_beam_idx].clone(),
                    next_score.item(),
                    beam_indices=beam_index,
                    generated_len=cur_len - decoder_prompt_len,
                    past_key_values=cur_past_key_values
                )
            else:
                # add next predicted token since it is not eos_token
                next_beam_scores[batch_idx, beam_idx] = next_score
                next_beam_tokens[batch_idx, beam_idx] = next_token
                next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                beam_idx += 1

            # once the beam for next step is full, don't add more tokens to it.
            if beam_idx == self.group_size:
                break

        if beam_idx < self.group_size:
            raise ValueError(
                f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
            )

        # Check if we are done so that we can save a pad step if all(done)
        self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
            next_scores[batch_idx].max().item(), cur_len, decoder_prompt_len
        )

    return UserDict(
        {
            "next_beam_scores": next_beam_scores.view(-1),
            "next_beam_tokens": next_beam_tokens.view(-1),
            "next_beam_indices": next_beam_indices.view(-1),
        }
    )

def beam_search_finalize(
    self,
    input_ids: torch.LongTensor,
    final_beam_scores: torch.FloatTensor,
    final_beam_tokens: torch.LongTensor,
    final_beam_indices: torch.LongTensor,
    max_length: int,
    pad_token_id: Optional[Union[int, torch.Tensor]] = None,
    eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None,
    beam_indices: Optional[torch.LongTensor] = None,
    decoder_prompt_len: Optional[int] = 0,
    past_key_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.LongTensor]:
    batch_size = len(self._beam_hyps) // self.num_beam_groups

    if eos_token_id is not None and not isinstance(eos_token_id, torch.Tensor):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id = torch.tensor(eos_token_id)

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
        if self._done[batch_group_idx]:
            continue

        # all open beam hypotheses are added to the beam hypothesis
        # beam hypothesis class automatically keeps the best beams
        for index_per_group in range(self.group_size):
            batch_beam_idx = batch_group_idx * self.group_size + index_per_group
            final_score = final_beam_scores[batch_beam_idx].item()
            final_tokens = input_ids[batch_beam_idx]
            beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
            generated_len = final_tokens.shape[-1] - decoder_prompt_len

            if past_key_values is not None:
                cur_past_key_values = DynamicCache(len(past_key_values))
                for i, (k, v) in enumerate(past_key_values):
                    cur_past_key_values.update(
                        k[batch_beam_idx : batch_beam_idx + 1], 
                        v[batch_beam_idx : batch_beam_idx + 1], 
                        i
                    )

            beam_hyp.add(
                final_tokens, 
                final_score, 
                beam_indices=beam_index, 
                generated_len=generated_len, 
                past_key_values=cur_past_key_values
            )

    # select the best hypotheses
    sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
    best = []
    best_indices = []
    best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
    best_kv_caches = []
    # retrieve best hypotheses
    for i in range(batch_size):
        beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups : (i + 1) * self.num_beam_groups]
        candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
        sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
        for j in range(self.num_beam_hyps_to_keep):
            best_hyp_tuple = sorted_hyps.pop()
            best_score = best_hyp_tuple[0]
            best_hyp = best_hyp_tuple[1]
            best_index = best_hyp_tuple[2]
            kv_cache = best_hyp_tuple[3]
            sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

            # append hyp to lists
            best.append(best_hyp)

            # append indices to list
            best_indices.append(best_index)

            best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
            best_kv_caches.append(kv_cache)
    
    # prepare for adding eos
    sent_lengths_max = sent_lengths.max().item() + 1
    sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
    decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

    if len(best_indices) > 0 and best_indices[0] is not None:
        indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
    else:
        indices = None

    # shorter batches are padded if needed
    if sent_lengths.min().item() != sent_lengths.max().item():
        if pad_token_id is None:
            raise ValueError("`pad_token_id` has to be defined")
        decoded.fill_(pad_token_id)

    if indices is not None:
        indices.fill_(-1)

    # fill with hypotheses and eos_token_id if the latter fits in
    for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
        decoded[i, : sent_lengths[i]] = hypo

        if indices is not None:
            indices[i, : len(best_idx)] = torch.tensor(best_idx)

        if sent_lengths[i] < sent_max_len:
            # inserting only the first eos_token_id
            decoded[i, sent_lengths[i]] = eos_token_id[0]

    return UserDict(
        {
            "sequences": decoded,
            "sequence_scores": best_scores,
            "beam_indices": indices,
            "past_key_values": best_kv_caches,
        }
    )


def beam_hypotheses_add(
    self,
    hyp: torch.LongTensor,
    sum_logprobs: float,
    beam_indices: Optional[torch.LongTensor] = None,
    generated_len: Optional[int] = None,
    past_key_values: Optional[torch.Tensor] = None,
):
    """
    Add a new hypothesis to the list.
    """
    if generated_len is not None:
        score = sum_logprobs / (generated_len**self.length_penalty)
    # This 'else' case exists for retrocompatibility
    else:
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)

    if len(self) < self.num_beams or score > self.worst_score:
        self.beams.append((score, hyp, beam_indices, past_key_values))
        if len(self) > self.num_beams:
            sorted_next_scores = sorted([(s, idx) for idx, (s, _, _, _) in enumerate(self.beams)])
            del self.beams[sorted_next_scores[0][1]]
            self.worst_score = sorted_next_scores[1][0]
        else:
            self.worst_score = min(score, self.worst_score)


@staticmethod
def generation_mixin_expand_inputs_for_generation(
    expand_size: int = 1,
    is_encoder_decoder: bool = False,
    input_ids: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
    # Do not call torch.repeat_interleave if expand_size is 1 because it clones
    # the input tensor and thus requires more memory although no change is applied
    if expand_size == 1:
        return input_ids, model_kwargs

    def _expand_dict_for_generation(dict_to_expand):
        for key in dict_to_expand:
            if (
                key != "cache_position"
                and dict_to_expand[key] is not None
                and isinstance(dict_to_expand[key], torch.Tensor)
            ):
                dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
        return dict_to_expand

    if input_ids is not None:
        input_ids = input_ids.repeat_interleave(expand_size, dim=0)

    model_kwargs = _expand_dict_for_generation(model_kwargs)
    if "past_key_values" in model_kwargs:
        for i, (k, v) in enumerate(model_kwargs["past_key_values"]):
            model_kwargs["past_key_values"].key_cache[i] = k.repeat_interleave(expand_size, dim=0)
            model_kwargs["past_key_values"].value_cache[i] = v.repeat_interleave(expand_size, dim=0)

    if is_encoder_decoder:
        if model_kwargs.get("encoder_outputs") is None:
            raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

    return input_ids, model_kwargs


@torch.no_grad()
def generation_mixin_generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    encoder_input_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config ([`~generation.GenerationConfig`], *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
            to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
            deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    # self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
        model_kwargs["num_logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
    cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
    user_defined_cache = model_kwargs.get(cache_name)
    max_cache_length = generation_config.max_length
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=encoder_input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache


    if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run beam sample
        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is not False  # Should check for `True` after v4.47
        and not is_torchdynamo_compiling()
        and hasattr(result, "past_key_values")
        and hasattr(result.past_key_values, "to_legacy_cache")
        and result.past_key_values.to_legacy_cache is not None
    ):
        # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
        should_convert_cache = generation_config.return_legacy_cache
        is_user_defined_cache = user_defined_cache is not None
        is_default_cache_type = (
            type(result.past_key_values) == DynamicCache  # noqa E721
            or (
                isinstance(result.past_key_values, EncoderDecoderCache)
                and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
                and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
            )
        )
        if not is_user_defined_cache and is_default_cache_type:
            logger.warning_once(
                "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                "keep returning the legacy format, please set `return_legacy_cache=True`."
            )
            should_convert_cache = True
        if should_convert_cache:
            result.past_key_values = result.past_key_values.to_legacy_cache()
    return result

def generation_mixin_beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`:
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    sequential = generation_config.low_memory
    do_sample = generation_config.do_sample

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False

    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        # if sequential is True, split the input to batches of batch_size and run sequentially
        if sequential:
            if any(
                model_name in self.__class__.__name__.lower()
                for model_name in [
                    "fsmt",
                    "reformer",
                    "ctrl",
                    "gpt_bigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                    "jamba",
                ]
            ):
                raise RuntimeError(
                    f"Currently generation for {self.__class__.__name__} is not supported "
                    f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                )

            inputs_per_sub_batches = _split_model_inputs(
                model_inputs,
                split_size=batch_size,
                full_batch_size=batch_beam_size,
                config=self.config.get_text_config(),
            )
            outputs_per_sub_batch = [
                self(**inputs_per_sub_batch, return_dict=True) for inputs_per_sub_batch in inputs_per_sub_batches
            ]

            outputs = stack_model_outputs(outputs_per_sub_batch, self.config.get_text_config())

        else:  # Unchanged original behavior
            outputs = self(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        # .float() is needed to retain precision for later logits manipulations
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
        # non eos token per beam.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)
        else:
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
            )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
            past_key_values=model_kwargs.get("past_key_values", None),
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
        # (that way the memory peak does not include outputs.logits)
        del outputs

        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
            this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
        past_key_values=model_kwargs.get("past_key_values", None),
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=sequence_outputs["past_key_values"],
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                logits=raw_logits,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=sequence_outputs["past_key_values"],
            )
    else:
        return sequence_outputs["sequences"]


def patch_hf():    
    # Patch beam search
    BeamSearchScorer.process = beam_search_process
    BeamSearchScorer.finalize = beam_search_finalize
    BeamHypotheses.add = beam_hypotheses_add
    # Patch generation mixin
    GenerationMixin.generate = generation_mixin_generate
    GenerationMixin._expand_inputs_for_generation = generation_mixin_expand_inputs_for_generation
    GenerationMixin._beam_search = generation_mixin_beam_search