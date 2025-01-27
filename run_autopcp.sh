export HYDRA_FULL_ERROR=1
# export SLURM_CLUSTER=false
# export USE_CPU=1

export USER=anything && python -m stopes.modules +compare_audios=AutoPCP_multilingual_v2 \
    +compare_audios.input_file=/asr4/anuprabha/is25/stopes/torgo_stops_differetext.tsv \
    ++compare_audios.src_audio_column=src_audio \
    ++compare_audios.tgt_audio_column=tgt_audio \
    +compare_audios.named_columns=true \
    +compare_audios.output_file=/asr4/anuprabha/is25/stopes/output_difftext.txt \
launcher=local
