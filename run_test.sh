code_dir=THUMT
work_dir=$PWD/dtmt_incredec
train_dir=$work_dir/data
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

for idx in `seq 1000 1000 1000` 
do
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/train/checkpoint
    cat $work_dir/train/checkpoint
    echo decoding with checkpoint-$idx
    python $work_dir/$code_dir/thumt/bin/translator.py \
        --models rnnsearch \
        --checkpoints $work_dir/train \
        --input $train_dir/valid_src \
        --output $train_dir/valid_src.out \
        --vocabulary $train_dir/vocab_32k.ch $train_dir/vocab_32k.en \
        --parameters=decode_batch_size=64
    echo evaluating with checkpoint-$idx
    cd $train_dir
    sh eval.sh valid_src.out.delbpe.eval.$idx
    cat valid_src.out.delbpe.eval.$idx
    cd $work_dir
    echo finished of checkpoint-$idx
done
