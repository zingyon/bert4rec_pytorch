from transformers import BertForMaskedLM
from typing import Optional, Union, Tuple
import torch
from torch.nn import CrossEntropyLoss


def get_mask_positions(input_ids, prob=0.2):
    
    mask = (torch.rand_like(input_ids, dtype=torch.float) < prob) # batch, seq_len
    
    while (mask.sum(dim=-1) == 0).any():
        mask = (torch.rand_like(input_ids, dtype=torch.float) < prob)
    
    return mask.to(input_ids.device)


class Bert4Rec(BertForMaskedLM):
    def __init__(self, config, masking_prob=0.2):
        super().__init__(config)
        self.masking_prob = masking_prob
        self.mask_id = config.vocab_size - 1

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if self.training:
            mask_position = get_mask_positions(input_ids, self.masking_prob)
            
            labels = input_ids.clone()
        else:
            input_ids[input_ids == -100] = self.mask_id
            mask_position = input_ids == self.mask_id
            
        
        masked_input_ids = input_ids.clone()
        
        labels[labels == 0] = -100
        
        masked_input_ids[mask_position] = self.mask_id

        outputs = self.bert(
            masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output[mask_position])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels[mask_position].view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
