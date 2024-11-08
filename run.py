from transformers import BertConfig, TrainingArguments, Trainer
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser

from models.model import Bert4Rec
from models.metric import compute_metric

from utils.preprocess import create_datasets, tuncate_and_padding

@dataclass
class DataArguments:
    max_seq_length: int=field(
        default=50, metadata={"help": "모델에 입력될 최대 시퀀스 길이"}
    )
    split: str=field(
        default="leave_one_out", metadata={"help": "데이터 스플릿 방식"} # 
    )
    input_files: str=field(
        default=None, metadata={"help": "데이터 원천 파일"}
    )
    min_rating: int=field(
        default=4, metadata={"help": "최소 평점"}
    )
    min_uc: int=field(
        default=10, metadata={"help": "유지할 유저의 최소 리뷰 수"}
    )
    min_sc: int=field(
        default=0, metadata={"help": "유지할 아이템의 최소 리뷰 수"}
    )
    
@dataclass
class ModelArguments:
    hidden_size: int=field(
        default=64, metadata={"help": "히든 사이즈"}
    )
    num_hidden_layers: int=field(
        default=2, metadata={"help": "인코더 레이어 수"}
    )
    num_attention_heads: int=field(
        default=2, metadata={"help": "멀티 헤드 어텐션 헤드 수"}
    )
    intermediate_size: int=field(
        default=256
    )
    hidden_dropout_prob: float=field(
        default=0.1
    # 드론아웃 확률 설정
    )
    attention_probs_dropout_prob: float=field(
        default=0.1
    # 드론아웃 확률 설정
    )
    layer_norm_eps: float=field(
        default=1e-12
    )
    

if __name__ == "__main__":
    
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()
    
    
    train_dataset, val_dataset, test_dataset, smap = create_datasets(data_args.input_files, data_args.min_rating, 
                                                               data_args.min_sc, data_args.min_uc, data_args.split)
    
    
    train_dataset = tuncate_and_padding(train_dataset, True, data_args.max_seq_length)
    val_dataset = tuncate_and_padding(val_dataset, False, data_args.max_seq_length)
    vocab_size = len(smap) + 2
    model_args = asdict(model_args)
    model_args["vocab_size"] = vocab_size
    
    config = BertConfig(**model_args)
    
    model = Bert4Rec(config)
    
    trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, 
                      eval_dataset=val_dataset,
                      compute_metrics=compute_metric)
    
    
    trainer.train()
    #option.
