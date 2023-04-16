use_mlqa=$1

if [ "$use_mlqa" = "use_mlqa" ]; then
    echo "Using MLQA dataset."
    python train_ctm.py \
    --do_eval_doc \
    --use_mlqa \
    --languages french vietnamese \
    --dataset data_mlqa/all_passages/ \
    --vocab_path save/models/mlqa_topic_models/french_vietnamese/ctm_new_vocab_20k.pkl \
    --data_preparation_file save/models/mlqa_topic_models/french_vietnamese/data_cache.pkl \
    --model_path_prefix save/models/mlqa_topic_models/french_vietnamese/ctm_20k_topics_ \
    --output_path save/models/mlqa_topic_models/french_vietnamese/ctm_20k_topics_NCLUSTER/DATASET_ctm_20k_topics_ \
    --sbert_name setu4993/LaBSE
    echo "Doing eval_doc"
    # echo "Doing both train; eval_doc"

# save/models/topic_models/ctm_20k_topics_4/cmu_ctm_20k_topics_4.npy
# save/models/topic_models/ctm_20k_topics_4/wow_ctm_20k_topics_4.npy
else 
    echo "Using knowexpert dataset."
    python train_ctm.py \
    --do_train \
    --do_eval_doc \
    --dataset data/wiki_articles.txt \
    --vocab_path save/models/topic_models/ctm_new_vocab_20k.pkl \
    --data_preparation_file save/models/topic_models/data_cache.pkl \
    --model_path_prefix save/models/topic_models/ctm_20k_topics_ \
    --output_path save/models/topic_models/ctm_20k_topics_NCLUSTER/DATASET_ctm_20k_topics_ \
    --sbert_name sentence-transformers/stsb-roberta-base-v2
fi

# # save/models/topic_models/ctm_20k_topics_4/cmu_ctm_20k_topics_4.npy
# # save/models/topic_models/ctm_20k_topics_4/wow_ctm_20k_topics_4.npy