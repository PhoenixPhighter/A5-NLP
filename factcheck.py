# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=1)

        del inputs, outputs, logits
        gc.collect()

        return probs[0][0] > .4


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    def predict(self, fact: str, passages: List[dict]) -> str:
        A, B = set(), set()
        for word in fact.split():
            for ele in word:
                if ele in self.punc:
                    word = word.replace(ele, "")
            if word.isalnum():
                A.add(word.lower())
        for p in passages:
            for word in p["text"].split():
                for ele in word:
                    if ele in self.punc:
                        word = word.replace(ele, "")
                if word.isalnum():
                    B.add(word.lower())
        recall = len(A.intersection(B)) / len(A)
        return "S" if recall > .75 else "NS"
        
    def predict_sentence(self, fact: str, sentence: str) -> str:
        A, B = set(), set()
        for word in fact.split():
            for ele in word:
                if ele in self.punc:
                    word = word.replace(ele, "")
            if word.isalnum():
                A.add(word.lower())
        for word in sentence.split():
            for ele in word:
                if ele in self.punc:
                    word = word.replace(ele, "")
            if word.isalnum():
                B.add(word.lower())
        recall = len(A.intersection(B)) / len(A)
        return "S" if recall > .125 else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.word_overlap = WordRecallThresholdFactChecker()

    def predict(self, fact: str, passages: List[dict]) -> str:
        
        for p in passages:
            for sentence in p["text"].split("."):
                pruned_text = sentence
                pruned_text = pruned_text.replace("<s>", "")
                pruned_text = pruned_text.replace("</s>", "")
                pruned_fact = fact
                if self.word_overlap.predict_sentence(pruned_fact, pruned_text) == "NS":
                    continue
                if self.ent_model.check_entailment(pruned_text, pruned_fact) == 1:
                    return "S"
        return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

