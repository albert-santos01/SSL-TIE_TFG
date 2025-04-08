python -u test_entire_models.py \
    --batch_size 128 \
    --n_threads 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --spec_DAVENet \
    --gpus 0 \
    --use_wandb \
    --project_wandb VG_SSL-TIE_test \
    --run_name lr1e-4-B128-S2M-SdT128\
    --exp_name SSL_TIE_PlacesAudio-lr1e-4-B128-S2M-SdT128 \
    --video \
    --val_video_idx 10 \
    --val_video_freq 1 \
    --cross_modal_freq 1 \
    --placesAudio $DATA/PlacesAudio_400k_distro/metadata/ \
    --simtype MISA \
    --links_path /home/asantos/models/SSL_TIE_PlacesAudio-lr1e-4-B128-SISA-SdT128/links_resume_SSL_TIE_PlacesAudio-lr1e-4-B128-SISA-SdT128_158491.json
    