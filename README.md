

```
python run_image_classification.py \
    --output_dir ./models/tcg_magic/ \
    --remove_unused_columns False \
    --image_column_name image \
    --label_column_name label \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 420 \
    --fp16 \
    --dataset_name acidtib/tcg-magic-cards \
    --push_to_hub \
    --push_to_hub_model_id tcg-magic-classifier
```

local dataset
```
--train_dir /media/acid/turtle/datasets/tcg_magic/data/train
--validation_dir /media/acid/turtle/datasets/tcg_magic/data/valid
```

huggingface dataset
```
--dataset_name acidtib/tcg-magic-cards
```

upload model to huggingface
```
--push_to_hub
--push_to_hub_model_id tcg-magic-classifier
```