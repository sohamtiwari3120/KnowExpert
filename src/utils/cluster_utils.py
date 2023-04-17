import os	
import sys	
import json	
import pickle	
import random	
import logging	
import time	
import argparse	
import numpy as np	
from tqdm import tqdm, trange	
from datetime import datetime

import torch	
from torch.utils.data import Dataset, DataLoader	

logger = logging.getLogger(__name__)

def load_mlqa_docs(data_path, languages):
    documents = []
    if "french" in languages:
        documents += json.load(open(os.path.join(data_path, "fr.json"), "r"))
    if "vietnamese" in languages:
        documents += json.load(open(os.path.join(data_path, "vi.json"), "r"))
    if "english" in languages:
        documents += json.load(open(os.path.join(data_path, "en.json"), "r"))
    if "chinese" in languages:
        documents += json.load(open(os.path.join(data_path, "zh.json"), "r"))
    return documents

def getClusters(doc_path, cluster_path, cluster_ind, data_dir="", \
    load_cmu=False, cmu_doc="", cmu_path="", topic_modeling=True, \
    load_topic=False, topic_path="", cmu_topic_path="", use_mlqa=False, languages=[]):

    # if not topic_modeling:
    #     logger.info("Using Topic Cluster...")
    #     cluster = TopicCluster(data_dir=data_dir)
    #     return cluster.get_topic_data(index=args.index)

    # Load the data
    if use_mlqa:
        wow_data = load_mlqa_docs(os.path.join(data_dir, "all_passages"), languages)
    else:
        with open(doc_path, "r") as f:
            wow_data = f.readlines()
        
    # load the topic if needed
    if load_topic:
        if not use_mlqa:
            with open(topic_path, "r") as f:
                topics = json.load(f)

    if load_cmu:
        with open(cmu_doc, "r") as f:
            cmu_data = f.readlines()

    if cluster_path is None or not topic_modeling:
        return wow_data + cmu_data

    # Filter data based on clusters
    logger.info("Using Topic Modeling...")
    with open(cluster_path, "rb") as f:
        cluster = np.load(f)
    cluster = list(cluster)

    assert len(wow_data) == len(cluster)
    selected = []
    selected_topics = []
    for ind, line in enumerate(wow_data):
        if np.argmax(cluster[ind]) == cluster_ind:
            selected.append(line.strip())
            if load_topic:
                if use_mlqa:
                    topic = line.split('//')[-1].strip().split('-')[-1].strip()
                    selected_topics.append(topic)
                else:
                    selected_topics.append(topics[ind])
    
    if load_cmu:
        with open(cmu_path, "rb") as f:
            cmu_inds = np.load(f)
        cmu_inds = list(cmu_inds)    

        if load_topic:
            with open(cmu_topic_path, "r") as f:
                topics = json.load(f)    
        
        assert len(cmu_data) == len(cmu_inds)
        for ind, line in enumerate(cmu_data):
            if cmu_inds[ind] == cluster_ind:
                selected.append(line.strip())
                if load_topic:
                    selected.append(topics[ind])
    logger.info(f"Number of Docs = {len(selected)}")
    return selected, selected_topics


def getWikiClusters(wiki_path, cluster_ind):
    with open(wiki_path.replace("NUM", str(cluster_ind)), "r") as f:
        lines = f.readlines()
    
    texts, topics = [], []
    for line in lines:
        items = line.strip().split("\t")
        topic, text = items[0], items[1]
        texts.append(text)
        topics.append(topic)
    
    return texts, topics