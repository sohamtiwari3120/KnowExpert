"""This script leverages the package [contextualized_topic_model](https://github.com/MilaNLProc/contextualized-topic-models) to build a contextualized topic model."""

import os
import string
import warnings
import argparse
import pickle as pkl
import numpy as np
import scipy
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation, bert_embeddings_from_list
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from sklearn.feature_extraction.text import CountVectorizer
from yaml import parse
import json

from src.data_utils.data_reader import load_wow_episodes
from src.data_utils.cmu_dog_reader import load_cmu_episodes


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

class Preprocessing(WhiteSpacePreprocessing):
    """
    Extended class of WhiteSpacePreprocessing so to handle test sets
    """
    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000):
        super().__init__(documents, stopwords_language, vocabulary_size)
    
    def preprocess(self):
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                                 for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary = set(vectorizer.get_feature_names_out())
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])
        self.vocab = list(vocabulary)
        return preprocessed_docs, unpreprocessed_docs, list(vocabulary)

    def preprocess_test(self, test_documents, vocab=None):
        if vocab is None:
            vocab = self.vocab
        preprocessed_docs_tmp = test_documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                                 for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b', vocabulary=vocab)
        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary = set(vectorizer.get_feature_names_out())
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            preprocessed_docs.append(doc)
            unpreprocessed_docs.append(self.documents[i])
        
        return preprocessed_docs, unpreprocessed_docs, list(vocabulary)


class DataPreparation(TopicModelDataPreparation):
    def __init__(self, contextualized_model=None):
        super().__init__(contextualized_model)
    def create_test_set(self, text_for_contextual, text_for_bow):
        # self.vectorizer = vectorizer
        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")
        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("The method did not have in input the text_for_bow parameter. This IS EXPECTED if you are using ZeroShotTM in a cross-lingual setting")
            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 20000)))
        
        test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model, batch_size=100, max_seq_length=512)
        return CTMDataset(test_contextualized_embeddings, test_bow_embeddings, self.id2token)
    
    def create_training_set(self, text_for_contextual, text_for_bow):

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        print(self.contextualized_model)
        train_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model, max_seq_length=512)
        self.vocab = self.vectorizer.get_feature_names_out()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        return CTMDataset(train_contextualized_embeddings, train_bow_embeddings, self.id2token)


def load_and_infer_docs(data_file, vocab_file, data_preparation_file, sbert_name, model_path_prefix, output_path, onehot=True, use_mlqa=False, languages=[]):
    # get topic distribution over text dataset
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if use_mlqa:
        documents = load_mlqa_docs(data_path=data_file, languages=languages)
    else:
        documents = [line.strip() for line in open(data_file, encoding="utf-8").readlines()]
    print(len(documents))
    with open(vocab_file, 'rb') as f:
        vocab = pkl.load(f)
    sp = Preprocessing(documents, "english", vocabulary_size=20000)
    preprocessed_documents_for_bow, unpreprocessed_corpus_for_contextual, _ = sp.preprocess_test(documents, vocab)
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    with open(data_preparation_file, 'rb') as f:
        tp = pkl.load(f)
    tp.contextualized_model = sbert_name
    testing_dataset = tp.transform(unpreprocessed_corpus_for_contextual, preprocessed_documents_for_bow)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path + '.pkl', 'wb') as f:
        pkl.dump(testing_dataset, f)
    for n_comp in [18, 9]:
        print(n_comp)
        ctm = CombinedTM(bow_size=20000, contextual_size=768, num_epochs=50, n_components=n_comp)
        ctm.num_data_loader_workers = 2
        base_dir = model_path_prefix + str(n_comp)
        model_details_folder = [entry for entry in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, entry))]
        assert len(model_details_folder) == 1, f"{model_details_folder}"
        ctm.load(model_path_prefix + str(n_comp) + f"/{model_details_folder[0]}", epoch=49)
        predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=10)
        if onehot:
            predictions = np.argmax(predictions, axis=1)
        save_path = output_path.replace("NCLUSTER", str(n_comp)) + str(n_comp) + '.npy'

        # ensure there's existing folder to save the results
        save_folder = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            np.save(f, predictions)


def load_and_infer_dialogue(dataset, split, vocab_file, data_preparation_file, sbert_name, model_path_prefix, output_path, onehot=False, add_response=True, use_mlqa=False, languages=[]):
    # get topic distribution over dialogue history
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if dataset == "wow":
        episodes = load_wow_episodes('./data', split, history_in_context=True, max_episode_length=1)
    elif dataset == "cmu_dog":
        episodes = load_cmu_episodes('./data', split)
    else:
        raise NotImplementedError
    contexts = []
    for episode in episodes:
        # if add_response:
        #     episode['text'].append(episode['response'])

        # tmp = ' '.join(episode['text'])
        contexts.append(episode['response'])
    with open(vocab_file, 'rb') as f:
        vocab = pkl.load(f)
    breakpoint()
    sp = Preprocessing(contexts, "english", vocabulary_size=20000)
    preprocessed_documents_for_bow, unpreprocessed_corpus_for_contextual, _ = sp.preprocess_test(contexts, vocab)
    with open(data_preparation_file, 'rb') as f:
        tp = pkl.load(f)
    tp.contextualized_model = sbert_name
    testing_dataset = tp.transform(unpreprocessed_corpus_for_contextual, preprocessed_documents_for_bow)
    for n_comp in [4, 8, 16]:
        ctm = CombinedTM(bow_size=20000, contextual_size=768, num_epochs=50, n_components=n_comp)
        ctm.load(model_path_prefix + str(n_comp), epoch=49)
        predictions = ctm.get_doc_topic_distribution(testing_dataset, n_samples=10)
        if onehot:
            predictions = np.argmax(predictions, axis=1)
        
        save_path = output_path.replace("NCLUSTER", str(n_comp)) + split +'_'+str(n_comp)+'.npy'
        # ensure there's existing folder to save the results
        save_folder = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_folder, exist_ok=True)

        with open(save_path, 'wb') as f:
            np.save(f, predictions)



def train(data_path, vocab_path, data_preparation_file, model_path_prefix, sbert_name="sentence-transformers/stsb-roberta-base-v2", use_mlqa=False, languages=[]):
    # train CTM on knowledge corpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if use_mlqa:
        documents = load_mlqa_docs(data_path, languages)        
    else:
        documents = [line.strip() for line in open(data_path, encoding="utf-8").readlines()]
    sp = Preprocessing(documents, "english", vocabulary_size=20000)
    preprocessed_documents_for_bow, unpreprocessed_corpus_for_contextual, vocab = sp.preprocess()
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pkl.dump(vocab, f)
    tp = DataPreparation(sbert_name)
    training_dataset = tp.create_training_set(unpreprocessed_corpus_for_contextual, preprocessed_documents_for_bow)
    os.makedirs(os.path.dirname(data_preparation_file), exist_ok=True)
    with open(data_preparation_file, 'wb') as f:
        pkl.dump(tp, f)
    for n_comp in [18, 9]:
        ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, num_epochs=50, n_components=n_comp)
        ctm.fit(training_dataset)
        print(ctm.get_topic_lists(5))

        # ensure there's existing folder to save the results
        save_folder = "/".join(model_path_prefix.split("/")[:-1])
        os.makedirs(save_folder, exist_ok=True)

        ctm.save(models_dir=model_path_prefix + str(n_comp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CTM / Infer CTM')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval_doc', action='store_true')
    parser.add_argument('--do_eval_dial', action='store_true')
    parser.add_argument('--use_mlqa', action='store_true')
    parser.add_argument("-l", '--languages', nargs='+', default=["english", "chinese", "french", "vietnamese"])

    parser.add_argument('--dataset', help='Path to knowledge corpus', type=str, default="data/wiki_articles.txt")
    parser.add_argument('--vocab_path', help='Vocabulary path (you can also build your own vocabulary)', type=str, default="save/models/topic_models/ctm_new_vocab_20k.pkl")
    parser.add_argument('--data_preparation_file', help='DataPreparation save path', type=str, required=True) 
    parser.add_argument('--model_path_prefix', help='Prefix of model save path', type=str, required=True)
    parser.add_argument('--output_path', help='Predicted topic distribution save path', type=str, required=True)
    parser.add_argument('--sbert_name', help='Name or path to sentence bert', type=str, required=False, default="sentence-transformers/stsb-roberta-base-v2")
    parser.add_argument('--onehot', help="onehot or not", type=bool, required=False)
    parser.add_argument('--hisres', help="add response into the context for topic prediction or not", action="store_true")
    
    args = parser.parse_args()
    if args.do_train:
        train(args.dataset, args.vocab_path, args.data_preparation_file, args.model_path_prefix, args.sbert_name, args.use_mlqa, args.languages)
    
    if args.do_eval_doc:
        if args.use_mlqa:
            load_and_infer_docs(args.dataset, args.vocab_path, args.data_preparation_file, args.sbert_name, args.model_path_prefix, args.output_path.replace("DATASET", "mlqa"), args.onehot, args.use_mlqa, args.languages)
        else:
            load_and_infer_docs(args.dataset, args.vocab_path, args.data_preparation_file, args.sbert_name, args.model_path_prefix, args.output_path.replace("DATASET", "wow"), args.onehot, args.use_mlqa, args.languages)
            load_and_infer_docs(args.dataset.replace("wiki", "cmu"), args.vocab_path, args.data_preparation_file, args.sbert_name, args.model_path_prefix, args.output_path.replace("DATASET", "cmu"), args.onehot, args.use_mlqa, args.languages)


    if args.do_eval_dial:
        if args.dataset == "cmu":
            splits = ["train", "valid", "test"]
        elif args.dataset == "wow":
            splits = ["train", "valid", "valid_unseen", "test", "test_unseen"]
        elif "mlqa" in args.dataset:
            splits = ["train", "val"]

        for split in splits:
            load_and_infer_dialogue(args.dataset, split, args.vocab_path, args.data_preparation_file, args.sbert_name, args.model_path_prefix, args.output_path, args.onehot, args.hisres, args.use_mlqa, args.languages)