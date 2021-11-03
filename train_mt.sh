USER_DIR='/home/drrndrrnprn/nlp/ABST/bartabst/'
DATA_PATH='/home/drrndrrnprn/nlp/ABST/datasets'
PREFIX='semeval-pengb'
DOMAIN='analyzed'

INPUT_PATH="$DATA_PATH/$PREFIX/$DOMAIN"
OUTPUT_PATH="$INPUT_PATH"

mkdir -p $OUTPUT_PATH

fairseq-train "$INPUT_PATH/data-bin" \
        --fp16 \
        --arch bart_base --layernorm-embedding \
        --task translation_from_pretrained_bart \
        --source-lang src \
        --target-lang tgt \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --optimizer adam \
        --adam-eps 1e-06 \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay \
        --lr 3e-05 \
        --stop-min-lr -1 \
        --warmup-updates 10000 \
        --total-num-update 40000 \
        --dropout 0.3 \
        --weight-decay 0.0 \
        --attention-dropout 0.1 \
        --clip-norm 0.1 \
        --max-tokens 4096 \
        --update-freq 2 \
        --save-interval 1 \
        --save-interval-updates 500 \
        --keep-interval-updates 10 \
        --no-epoch-checkpoints \
        --seed 222 \
        --log-format simple \
        --log-interval 20 \
        --restore-file "$DATA_PATH/bart.base/model.pt" \
        --reset-optimizer \
        --reset-meters \
        --reset-dataloader \
        --reset-lr-scheduler \
        --ddp-backend=no_c10d \
        --skip-invalid-size-inputs-valid-test \
        --save-dir "$OUTPUT_PATH/checkpoints" \
        --user-dir "$USER_DIR" \
        --tensorboard-logdir "$OUTPUT_PATH/tensorboard" | tee -a "$OUTPUT_PATH train.log" 
        