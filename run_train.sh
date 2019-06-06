code_dir=THUMT
work_dir=$PWD/dtmt_incredec
train_dir=$work_dir/data
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model rnnsearch \
  --output $work_dir/train \
  --input $train_dir/train_bpe30k.ch $train_dir/train_bpe30k.en \
  --vocabulary $train_dir/vocab.ch $train_dir/vocab.en \
  --validation $train_dir/valid_src \
  --references $train_dir/valid_trg_low.BPE.0 $train_dir/valid_trg_low.BPE.1 $train_dir/valid_trg_low.BPE.2 $train_dir/valid_trg_low.BPE.3 \
  --parameters=device_list=[0,1],eval_steps=10000000,train_steps=120000,batch_size=4096,max_length=128,constant_batch_size=False,embedding_size=1024,learning_rate=8e-4,learning_rate_decay=rnnplus_warmup_decay,warmup_steps=50,s=8000,e=100000,adam_epsilon=1e-6

