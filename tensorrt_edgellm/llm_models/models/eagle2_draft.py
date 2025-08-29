"""
EAGLE2 Draft Model Implementation

This module provides the EAGLE2 draft model implementation for efficient
accelerated generation. The draft model is used in speculative decoding
to predict multiple tokens ahead.

The module contains:
- Eagle2DraftModel: EAGLE2 draft model class with decoder layers
"""

import os
from typing import Any, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig

from ..layers.layers import EdgeLLMDecoderLayer, PromptTuningEmbedding
from .llm_model import EdgeLLMModelForCausalLM


class Eagle2DraftModel(nn.Module):
    """
    EAGLE2 Draft Model for speculative decoding.
    
    This model implements the draft component of EAGLE2, which predicts
    multiple tokens ahead to accelerate generation. It combines input
    embeddings with hidden states from the base model and processes them
    through decoder layers.
    
    Attributes:
        vocab_size: Size of the vocabulary
        draft_vocab_size: Size of the draft vocabulary (may differ from vocab_size)
        fc: Fusion layer that combines input embeddings and hidden states
        layers: List of decoder layers
        lm_head: Language model head for token prediction
    """

    def __init__(self, config: Any, use_prompt_tuning: bool = False) -> None:
        """
        Initialize the EAGLE2 draft model.
        
        Args:
            config: Model configuration object containing model parameters
        """
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size",
                                        config.vocab_size)
        self.use_prompt_tuning = use_prompt_tuning
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         padding_idx=config.pad_token_id)

        # Fusion layer to combine input embeddings and hidden states
        self.fc = nn.Linear(2 * config.hidden_size,
                            config.hidden_size,
                            bias=config.fc_bias)

        # Decoder layers using our custom EdgeLLMDecoderLayer with config
        self.layers = nn.ModuleList([
            EdgeLLMDecoderLayer(config, index, eagle3_draft=False)
            for index in range(config.num_hidden_layers)
        ])

        self.softmax = nn.Softmax(dim=-1)

        # Language model head for token prediction
        self.lm_head = nn.Linear(config.hidden_size,
                                 self.draft_vocab_size,
                                 bias=False)

    @property
    def device(self):
        """Get the device of the model's parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        past_key_values: List[torch.FloatTensor],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        hidden_states_from_base: torch.Tensor,
        hidden_states_from_draft: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the EAGLE2 draft model.
        
        Args:
            past_key_values: Past key-value cache for efficient decoding
                           List of tensors, each with shape (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings, shape (batch_size, seq_len, head_dim)
            context_lengths: Context length tensor indicating current position in cache, shape (batch_size,)
            last_token_ids: Indices of the last tokens to extract, shape (batch_size,)
            hidden_states_from_base: Hidden states from the base model, shape (batch_size, seq_len, hidden_size)
            hidden_states_from_draft: Hidden states from previous draft predictions, shape (batch_size, seq_len, hidden_size)
            position_ids: Position IDs for positional encoding, shape (batch_size, seq_len)
            attention_mask: Attention mask for the decoder layers, shape (batch_size, seq_len, seq_len + past_len)
            input_ids: Input token IDs of shape (batch_size, seq_len), optional (used for standard models and prompt tuning)
            image_embeds: Image embeddings tensor of shape (image_token_len, hidden_size), optional (used with prompt tuning)
            inputs_embeds: Input embeddings tensor of shape (batch_size, seq_len, hidden_size), optional (legacy support)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]: (logits, hidden_states, present_key_values)
                - logits: Predicted token logits, shape (batch_size, num_tokens, draft_vocab_size)
                - hidden_states: Final hidden states, shape (batch_size, num_tokens, hidden_size)
                - present_key_values: Updated key-value cache, tuple of tensors
        """

        # Handle input embeddings
        if inputs_embeds is None:
            if self.use_prompt_tuning:
                # For prompt tuning models, use prompt_tuning_embedding. Cannot make this member variable for load_state_dict issue.
                inputs_embeds = PromptTuningEmbedding(self.embed_tokens)(
                    input_ids, image_embeds)
            else:
                # For standard models, use embed_tokens
                inputs_embeds = self.embed_tokens(input_ids)

        # Combine hidden states from base model and draft
        hidden_states = hidden_states_from_base + hidden_states_from_draft

        # Fuse input embeddings with hidden states
        hidden_states = self.fc(
            torch.cat((inputs_embeds, hidden_states), dim=-1))

        present_key_values = ()

        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # Get the past_key_value for this specific layer
            past_key_value = past_key_values[idx] if isinstance(
                past_key_values, (list, tuple)) else past_key_values

            hidden_states, present_key_value = decoder_layer(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                context_lengths=context_lengths,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            present_key_values += (present_key_value, )

        # TODO: EAGLE Draft model uses an implicit remove_padding here to flatten the hidden states
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        # Extract last token hidden states and compute logits
        hidden_states = hidden_states[last_token_ids, :]
        logits = self.lm_head(hidden_states)
        logits = logits.to(torch.float32)
        # TODO: Remove this softmax
        logits = self.softmax(logits)

        return logits, hidden_states, present_key_values

    @classmethod
    def from_pretrained(
        cls,
        draft_model_dir: str,
        base_model: EdgeLLMModelForCausalLM,
        use_prompt_tuning: bool = False,
        max_position_embeddings: int = 4096,
    ) -> "Eagle2DraftModel":
        """
        Load a pre-trained EAGLE2 draft model.
        
        Args:
            draft_model_dir: Path to the draft model directory
            base_model: Base model to copy weights from if needed
            use_prompt_tuning: Whether to enable prompt tuning support
            max_position_embeddings: Maximum positional embedding length to use for model initialization
            
        Returns:
            Eagle2DraftModel: Loaded EAGLE2 draft model instance
            
        Raises:
            FileNotFoundError: If model files cannot be found
        """

        # Load configuration
        config = AutoConfig.from_pretrained(draft_model_dir)

        if use_prompt_tuning:
            if hasattr(config, 'text_config'):
                config = config.text_config

        # Hard overwrite the config max_position_embeddings
        print(
            f"Setting draft model max_position_embeddings to {max_position_embeddings}"
        )
        config.max_position_embeddings = max_position_embeddings

        pytorch_bin_path = os.path.join(draft_model_dir, "pytorch_model.bin")
        safetensors_path = os.path.join(draft_model_dir, "model.safetensors")
        if not os.path.exists(pytorch_bin_path):
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(
                    f"Model file not found at {pytorch_bin_path} or {safetensors_path}"
                )
            draft_state_dict = load_file(safetensors_path,
                                         device=str(base_model.device))
        else:
            draft_state_dict = torch.load(pytorch_bin_path,
                                          weights_only=True,
                                          map_location=base_model.device)

        # WARNING: Some EAGLE2 draft models have fc.bias or input_layernorm, some don't.
        if "fc.bias" in draft_state_dict:
            print("EAGLE2 draft model has fc.bias")
            config.fc_bias = True
        else:
            config.fc_bias = False
        if "layers.0.input_layernorm.weight" in draft_state_dict:
            print("EAGLE2 draft model has input_layernorm")
            config.input_layernorm = True
        else:
            config.input_layernorm = False

        model = cls(config, use_prompt_tuning=use_prompt_tuning)

        # Use weights from base model if missing
        if base_model is not None:
            base_state_dict = base_model.state_dict()
            # Use embed_tokens.weight from base model if missing
            if "embed_tokens.weight" not in draft_state_dict:
                if "embed_tokens.weight" in base_state_dict:
                    print("Using embed_tokens.weight from base model")
                    draft_state_dict["embed_tokens.weight"] = base_state_dict[
                        "embed_tokens.weight"]
                elif "model.embed_tokens.weight" in base_state_dict:
                    print("Using model.embed_tokens.weight from base model")
                    draft_state_dict["embed_tokens.weight"] = base_state_dict[
                        "model.embed_tokens.weight"]
                else:
                    raise ValueError(
                        "embed_tokens.weight not found in base model")

            # Use lm_head.weight from base model if missing
            if "lm_head.weight" not in draft_state_dict and "lm_head.weight" in base_state_dict:
                print("Using lm_head.weight from base model")
                draft_state_dict["lm_head.weight"] = base_state_dict[
                    "lm_head.weight"]

        model.load_state_dict(draft_state_dict)
        return model
