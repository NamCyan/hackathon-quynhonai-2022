from turtle import shape
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.modeling_outputs import SequenceClassifierOutput
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import numpy as np

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaMultiHeadClassifier(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_aspect = 6
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifiers = nn.ModuleList(RobertaClassificationHead(config) for i in range(self.num_aspect))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        all_logits = []
        # print(labels.shape, type(labels), "*"*100)
        # print(labels)

        for i in range(self.num_aspect):
            logits_ = self.classifiers[i](sequence_output)
            all_logits.append(logits_)
        logits = torch.cat(all_logits, dim= 0)

        loss = None
        if labels is not None:
            labels = torch.transpose(labels, 0, 1).contiguous().view(-1)
            # print(labels.view(-1))
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def MC_predict(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_sample: int = 10,
        majority_vote: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        self.eval()

        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = 0 if not majority_vote else []
        # print(labels.shape, type(labels), "*"*100)
        # print(labels)

        for j in range(num_sample):


            all_logits = []
            for i in range(self.num_aspect):
                logits_ = self.classifiers[i](sequence_output)
                all_logits.append(logits_)
            if majority_vote:
                logits.append(torch.cat(all_logits, dim= 0))
            else:
                logits += (torch.cat(all_logits, dim= 0) /num_sample)

        if majority_vote:
            logits = [torch.argmax(logit, dim=-1).unsqueeze(0) for logit in logits]
            logits = torch.cat(logits, dim=0)
            logits = torch.mode(logits, 0)[0]

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class RobertaMultiLSTMClassifier(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_aspect = 6
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lstm = nn.LSTM(input_size= config.hidden_size, hidden_size= config.hidden_size, num_layers=2, dropout= 0.2, batch_first=True, bidirectional= True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.drop = nn.Dropout(classifier_dropout)
        self.classifiers = nn.ModuleList(RobertaClassificationHead(config) for i in range(self.num_aspect))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output, (hn, cn) = self.lstm(sequence_output)
        sequence_output = self.drop(sequence_output[:,:,:self.config.hidden_size] + sequence_output[:,:,self.config.hidden_size:])
        
        all_logits = []
        # print(labels.shape, type(labels), "*"*100)
        # print(labels)

        for i in range(self.num_aspect):
            logits_ = self.classifiers[i](sequence_output)
            all_logits.append(logits_)
        logits = torch.cat(all_logits, dim= 0)

        loss = None
        if labels is not None:
            labels = torch.transpose(labels, 0, 1).contiguous().view(-1)
            # print(labels.view(-1))
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaAspectEmbedding(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_aspect = 6
        self.aspect_emd_dim = config.hidden_size
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        
        self.classifier = nn.Sequential(nn.Dropout(classifier_dropout),
                                      nn.Linear(config.hidden_size, config.hidden_size),
                                      nn.Tanh(),
                                      nn.Dropout(classifier_dropout),
                                      nn.Linear(config.hidden_size, config.num_labels))

        self.aspectEmbedding = nn.Embedding(self.num_aspect, self.aspect_emd_dim)
        # self.attention_transform = nn.Linear(config.hidden_size, self.aspect_emd_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # attention_transform = torch.tanh(self.attention_transform(sequence_output))

        aspectEmbedding = self.aspectEmbedding(torch.tensor(range(self.num_aspect)).cuda()).transpose(0,1)
        aspectEmbedding = aspectEmbedding.repeat(sequence_output.size(0),1,1)

        attention_score = F.softmax(torch.matmul(sequence_output, aspectEmbedding), dim=1)
        # print(attention_score)
        attention_score = torch.transpose(attention_score,1,2).unsqueeze(-1)
        # print(sequence_output)
        value = sequence_output.unsqueeze(1).repeat(1,self.num_aspect,1,1)*attention_score
        value = value.sum(2).view(-1, value.size(-1))
    

        # print(torch.bmm(sequence_output,torch.transpose(aspectEmbedding, 0, 1)).shape)

        logits = self.classifier(value)

        loss = None
        if labels is not None:
            labels = labels.view(-1)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HSUM(nn.Module):
    def __init__(self, count, config):
        super(HSUM, self).__init__()
        self.config = config
        self.count = count
        self.pre_layers = torch.nn.ModuleList()
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, layers, attention_mask, return_list= False):
        logitses = []
        output = torch.zeros_like(layers[0])

        for i in range(self.count):
            output = output + layers[-i-1]
            output = self.pre_layers[i](output, attention_mask)[0]
            logits = output 
            logitses.append(logits)

        if return_list:
            return logitses

        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return avg_logits

class PSUM(nn.Module):
    def __init__(self, count, config):
        super(PSUM, self).__init__()
        self.config = config
        self.count = count
        self.pre_layers = torch.nn.ModuleList()
        for i in range(count):
            self.pre_layers.append(RobertaLayer(config))
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, layers, attention_mask, return_list= False):
        logitses = []
        # output = torch.zeros_like(layers[0])

        for i in range(self.count):
            output = self.pre_layers[i](layers[-i-1], attention_mask)[0]
            logits = output 
            logitses.append(logits)

        if return_list:
            return logitses

        avg_logits = torch.sum(torch.stack(logitses), dim=0)/self.count
        return avg_logits


class RobertaMixLayer(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_aspect = 6
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.mixlayer = PSUM(4, config)
        self.classifiers = nn.ModuleList(RobertaClassificationHead(config) for i in range(self.num_aspect))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states= True,
            return_dict=True,
        )
        layers = outputs.hidden_states

        extend_attention_mask = (1.0 - attention_mask[:,None, None, :]) * -10000.0
        sequence_output = self.mixlayer(layers, extend_attention_mask)

        all_logits = []
        # print(labels.shape, type(labels), "*"*100)
        # print(labels)

        for i in range(self.num_aspect):
            logits_ = self.classifiers[i](sequence_output)
            all_logits.append(logits_)
        logits = torch.cat(all_logits, dim= 0)

        loss = None
        if labels is not None:
            labels = torch.transpose(labels, 0, 1).contiguous().view(-1)
            # print(labels.view(-1))
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaEnsembleLayer(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_aspect = 6
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.mixlayer = PSUM(4, config)
        self.classifiers = nn.ModuleList(RobertaClassificationHead(config) for i in range(self.num_aspect))

        self.loss_weight = torch.tensor([0.21247951408810348,
                                        4.032710224014733,
                                        4.013074249497874,
                                        3.8977634029868797,
                                        3.0683040463018503,
                                        2.4210085867882443]).cuda()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states= True,
            return_dict=True,
        )
        layers = outputs.hidden_states

        extend_attention_mask = (1.0 - attention_mask[:,None, None, :]) * -10000.0
        sequence_outputs = self.mixlayer(layers, extend_attention_mask, True)

        logits = 0
        # print(labels.shape, type(labels), "*"*100)
        # print(labels)

        for sequence_output in sequence_outputs:
            all_logit = []
            for i in range(self.num_aspect):
                logits_ = self.classifiers[i](sequence_output)
                all_logit.append(logits_)
            logits += torch.cat(all_logit, dim= 0)/len(sequence_outputs)

        loss = None
        if labels is not None:
            labels = torch.transpose(labels, 0, 1).contiguous().view(-1)
            # print(labels.view(-1))
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight= self.loss_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RACSA(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_aspect = 6
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.aspect_emb = RobertaModel(config.aspect_emb_config, add_pooling_layer=False)
        self.aspect_transform = nn.Linear(config.aspect_emb_config.hidden_size, config.hidden_size)


        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.classifiers = nn.ModuleList(nn.Sequential(nn.Dropout(classifier_dropout),
                                        nn.Linear(config.hidden_size*2, config.hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(classifier_dropout),
                                        nn.Linear(config.hidden_size, config.num_labels)) for i in range(self.num_aspect))

        self.doc_attention = nn.Linear(config.hidden_size, config.hidden_size)
        self.context_vector = nn.Linear(config.hidden_size, 1, bias= False)

        self.loss_weight = torch.tensor([0.21247951408810348,
                                        4.032710224014733,
                                        4.013074249497874,
                                        3.8977634029868797,
                                        3.0683040463018503,
                                        2.4210085867882443]).cuda()
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states= output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        aspect_output = torch.tanh(self.aspect_transform(self.aspect_emb(**self.config.aspect_tokenized)[0]))

        doc_transform = torch.tanh(self.doc_attention(sequence_output))
        attention_score = torch.softmax(self.context_vector(doc_transform), dim= 1)
        doc_embedding = torch.sum(sequence_output * attention_score, dim= 1)

        all_logits= []
        for i in range(self.num_aspect):
            attention_score = torch.softmax(torch.max(torch.matmul(sequence_output, aspect_output[i].transpose(0,1)), dim=-1).values, dim=-1).unsqueeze(-1)
            
            sentence_embedding = torch.sum(sequence_output * attention_score, dim= 1)
            final_embeding = torch.cat([doc_embedding, sentence_embedding], dim= 1)

            logits_ = self.classifiers[i](final_embeding)
            all_logits.append(logits_)

        logits = torch.cat(all_logits, dim= 0)

        loss = None
        if labels is not None:
            labels = torch.transpose(labels, 0, 1).contiguous().view(-1)
            # print(labels.view(-1))
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.loss_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





if __name__ == "__main__":

    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(
        "../QuiNhonAI/vi-roberta-large-multi",
        num_labels = 6,
    )
    config.aspect_emb_config = AutoConfig.from_pretrained(
        "../QuiNhonAI/vi-roberta-part1",
        num_labels = 6,
    )
    a = RACSA.from_pretrained(
        "../QuiNhonAI/vi-roberta-large-multi",
        config = config,
    )
    a.cuda()

    ASPECTS = ["Dịch vụ vui chơi giải trí",
    "Dịch vụ lưu trú",
    "Hệ thống nhà hàng phục vụ khách du lịch",
    "Dịch vụ ăn uống",
    "Dịch vụ di chuyển",
    "Dịch vụ mua sắm"]

    tokenizer = AutoTokenizer.from_pretrained("../QuiNhonAI/vi-roberta-part1")
    config.aspect_tokenized = tokenizer(ASPECTS, padding="longest", max_length=20, truncation=True)
    # config.aspect_tokenized.set_format("torch")
    config.aspect_tokenized = {k: torch.tensor(v).to("cuda") for k, v in config.aspect_tokenized.items()}
    print(config.aspect_tokenized)

    inputs = tokenizer(["Xin chao cac ban", "Toi yeu em"], padding="longest", max_length=256, truncation=True)
    # inputs.set_format("torch")
    inputs = {k: torch.tensor(v).to("cuda") for k, v in inputs.items()}
    # print(inputs)
    b = a(**inputs, return_dict= True)
    print(b.logits.shape)
