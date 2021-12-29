from typing import Dict, List, Any
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from utils.detector import SimpleDetector
from model.multiatt.BilinearMatrixAttention import  BilinearMatrixAttention
from model.multiatt.masked_softmax import masked_softmax
from model.multiatt.InputVariationalDropout import InputVariationalDropout
import paddlenlp


class AttentionQA(nn.Layer):
    def __init__(self,

                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 ):
        super(AttentionQA, self).__init__()
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################
        #self.rnn_input_dropout = InputVariationalDropout(input_dropout,1)
        self.rnn_input_dropout = paddle.nn.Dropout(input_dropout)
        self.span_encoder = paddle.nn.LSTM(1280, 256, time_major=False,direction='bidirectional',weight_ih_attr=paddle.nn.initializer.XavierUniform(),weight_hh_attr=True)
        self.reasoning_encoder = paddle.nn.LSTM(1536, 256, num_layers =2,time_major=False,direction='bidirectional',weight_ih_attr=paddle.nn.initializer.XavierUniform(),weight_hh_attr=True)

        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512
        )
        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=self.detector.final_dim,
        )

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        dim = sum([d for d, to_pool in [(512, self.pool_reasoning),
                                        (512, self.pool_answer),
                                        (512, self.pool_question)] if to_pool])

        self.final_mlp = paddle.nn.Sequential(
            paddle.nn.Dropout(input_dropout),
            paddle.nn.Linear(dim, hidden_dim_maxpool,weight_attr=paddle.nn.initializer.XavierUniform()),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(input_dropout),
            paddle.nn.Linear(hidden_dim_maxpool, 1,weight_attr=paddle.nn.initializer.XavierUniform()),
        )
        self._accuracy =  paddle.metric.Accuracy()
        self._loss = paddle.nn.CrossEntropyLoss()

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = paddle.clip(span_tags, min=0)  # In case there were masked values here
        #row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id = paddle.zeros_like(span_tags_fixed)
        row_id_broadcaster = paddle.arange(0, row_id.shape[0], step=1)[:, None]
        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.reshape([-1]).astype('int32'), span_tags_fixed.reshape([-1]).astype('int32')].reshape([*span_tags_fixed.shape, -1])

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)
        span_rep = paddle.concat((span, retrieved_feats), -1)
        # add recurrent dropout here
        span_rep = span_rep.reshape([span_rep.shape[0]*span_rep.shape[1],span_rep.shape[2],-1])
        if self.rnn_input_dropout:
            ones = paddle.ones([span_rep.shape[0],span_rep.shape[-1]],dtype=span_rep.dtype)
            span_rep = self.rnn_input_dropout(ones).unsqueeze(1) * span_rep
        span_mask = paddle.sum(span_mask,-1).astype('int32').reshape([span_mask.shape[0]*span_mask.shape[1],1])
        repation,_ = self.span_encoder(span_rep,sequence_length = span_mask)
        return repation, retrieved_feats

    # def forward(self,
    #             images,
    #             objects,
    #             segms,
    #             boxes,
    #             box_mask,
    #             question,
    #             question_tags,
    #             question_mask,
    #             answers,
    #             answer_tags,
    #             answer_mask,
    #             label ):
    def forward(self,data):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        question=data[0]
        question_tags=data[1]
        answers=data[2]
        answer_tags=data[3]
        label=data[4]
        segms=data[5]
        objects=data[6]
        boxes=data[7]
        question_mask=data[8]
        answer_mask=data[9]
        box_mask=data[10]
        images=data[11]
        batch_size = question_mask.shape[0]
        # max_len = int(box_mask.sum(1).max().item())
        # objects = objects[:, :max_len]
        # box_mask = box_mask[:, :max_len]
        # boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]

        # for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
        #     if int(the_tags.max()) > max_len:
        #         raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
        #             tag_type, int(the_tags.max()), objects.shape, the_tags
        #         ))
        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        # Now get the question representations
        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        # Perform Q by A attention
        # [batch_size, 4, question_length, answer_length]
        qa_similarity = self.span_attention(q_rep,a_rep,
        ).reshape([batch_size, int(q_rep.shape[0]/batch_size), q_rep.shape[1], a_rep.shape[1]])
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        q_rep = q_rep.reshape([batch_size,int(q_rep.shape[0]/batch_size),q_rep.shape[1],-1])
        attended_q = paddlenlp.ops.einsum("bnqa,bnqd->bnad", (qa_attention_weights, q_rep))
        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.reshape([batch_size,int(a_rep.shape[0]/batch_size * a_rep.shape[1]), -1]),
                                             obj_reps['obj_reps']).reshape([batch_size, int(a_rep.shape[0]/batch_size),
                                                                        a_rep.shape[1], obj_reps['obj_reps'].shape[1]])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:, None, None] )
        attended_o = paddle.einsum("bnao,bod->bnad", atoo_attention_weights, obj_reps['obj_reps'])

        a_rep = a_rep.reshape([batch_size,int(a_rep.shape[0]/batch_size),a_rep.shape[1],-1])


        reasoning_inp = paddle.concat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
                                                           (attended_o, self.reasoning_use_obj),
                                                           (attended_q, self.reasoning_use_question)] if to_pool] , -1)

        reasoning_inp = reasoning_inp.reshape([reasoning_inp.shape[0]*reasoning_inp.shape[1],reasoning_inp.shape[2],-1])
        if self.rnn_input_dropout is not None:
            ones = paddle.ones([reasoning_inp.shape[0], reasoning_inp.shape[-1]], dtype=reasoning_inp.dtype)
            reasoning_inp = self.rnn_input_dropout(ones).unsqueeze(1) * reasoning_inp

        reasoning_output ,_= self.reasoning_encoder(reasoning_inp,
                                                  sequence_length =paddle.sum(answer_mask,-1).reshape([answer_mask.shape[0]*answer_mask.shape[1],1]))

        reasoning_output = reasoning_output.reshape([batch_size,int(reasoning_output.shape[0]/batch_size),reasoning_output.shape[1],reasoning_output.shape[2]])
        things_to_pool = paddle.concat([x for x, to_pool in [(reasoning_output , self.pool_reasoning),
                                                         (a_rep , self.pool_answer),
                                                         (attended_q , self.pool_question)] if to_pool], -1)
        pooled_rep = things_to_pool * answer_mask[..., None] + (1 - answer_mask[..., None]) * -1e7
        pooled_rep = pooled_rep.max(2)
        logits = self.final_mlp(pooled_rep)
        logits = logits.squeeze(2)
        ###########################################
        class_probabilities = F.softmax(logits, axis=-1)
        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'].reshape([-1]),
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        if label is not None:
            #print('---------------------------label.astype(int64)',label.astype('int64').reshape([-1]))
            # print('-----------------logits', logits)
            # print('-----------------label', label)
            loss = self._loss(logits, label.astype('int64').reshape([-1]))
            correct = self._accuracy.compute(logits, label)
            self._accuracy.update(correct)
            # print('++++++++++loss[None]',loss[None])
            output_dict["loss"] = loss[None].reshape([-1])

        return output_dict
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset:
            self._accuracy.reset()
        return {'accuracy': self._accuracy.accumulate()}































