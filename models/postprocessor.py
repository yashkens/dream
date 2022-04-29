from typing import Sequence, List, Tuple, Callable, Dict
import random
import itertools
import copy
import re

from core.state_schema import Dialog


def detokenize(tokens):
    """
    Detokenizing a text undoes the tokenizing operation, restores
    punctuation and spaces to the places that people expect them to be.
    Ideally, `detokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = " ".join(tokens)
    step0 = text.replace(". . .", "...")
    step1 = step0.replace("`` ", '"').replace(" ''", '"')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(" nt", "nt").replace("can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


class PersonNormalizer:
    """
    Detects mentions of mate user's name and either
    (0) converts them to user's name taken from state
    (1) either removes them.

    Parameters:
        person_tag: tag name that corresponds to a person entity
    """

    def __init__(self, person_tag: str = "PER", **kwargs):
        self.per_tag = person_tag

    def __call__(
        self, tokens: List[List[str]], tags: List[List[str]], names: List[str]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        out_tokens, out_tags = [], []
        for u_name, u_toks, u_tags in zip(names, tokens, tags):
            u_toks, u_tags = self.tag_mate_gooser_name(u_toks, u_tags, person_tag=self.per_tag)
            if u_name:
                u_toks, u_tags = self.replace_mate_gooser_name(u_toks, u_tags, u_name)
                if random.random() < 0.5:
                    u_toks = [u_name, ","] + u_toks
                    u_tags = ["B-MATE-GOOSER", "O"] + u_tags

                    u_toks[0] = u_toks[0][0].upper() + u_toks[0][1:]
                    if u_tags[2] == "O":
                        u_toks[2] = u_toks[2][0].lower() + u_toks[2][1:]
            else:
                u_toks, u_tags = self.remove_mate_gooser_name(u_toks, u_tags)
            out_tokens.append(u_toks)
            out_tags.append(u_tags)
        return out_tokens, out_tags

    @staticmethod
    def tag_mate_gooser_name(
        tokens: List[str], tags: List[str], person_tag: str = "PER", mate_tag: str = "MATE-GOOSER"
    ) -> Tuple[List[str], List[str]]:
        if "B-" + person_tag not in tags:
            return tokens, tags
        out_tags = []
        i = 0
        while i < len(tokens):
            tok, tag = tokens[i], tags[i]
            if i + 1 < len(tokens):
                if (tok == ",") and (tags[i + 1] == "B-" + person_tag):
                    # it might be mate gooser name
                    out_tags.append(tag)
                    j = 1
                    while (i + j < len(tokens)) and (tags[i + j][2:] == person_tag):
                        j += 1
                    if (i + j == len(tokens)) or (tokens[i + j][0] in ",.!?;)"):
                        # it is mate gooser
                        out_tags.extend([t[:2] + mate_tag for t in tags[i + 1 : i + j]])
                    else:
                        # it isn't
                        out_tags.extend(tags[i + 1 : i + j])
                    i += j
                    continue
            if i > 0:
                if (tok == ",") and (tags[i - 1][2:] == person_tag):
                    # it might have been mate gooser name
                    j = 1
                    while (len(out_tags) >= j) and (out_tags[-j][2:] == person_tag):
                        j += 1
                    if (len(out_tags) < j) or (tokens[i - j][-1] in ",.!?("):
                        # it was mate gooser
                        for k in range(j - 1):
                            out_tags[-k - 1] = out_tags[-k - 1][:2] + mate_tag
                    out_tags.append(tag)
                    i += 1
                    continue
            out_tags.append(tag)
            i += 1
        return tokens, out_tags

    @staticmethod
    def replace_mate_gooser_name(
        tokens: List[str], tags: List[str], replacement: str, mate_tag: str = "MATE-GOOSER"
    ) -> Tuple[List[str], List[str]]:
        assert len(tokens) == len(tags), f"tokens({tokens}) and tags({tags}) should have the same length"
        if "B-" + mate_tag not in tags:
            return tokens, tags

        repl_tokens = replacement.split()
        repl_tags = ["B-" + mate_tag] + ["I-" + mate_tag] * (len(repl_tokens) - 1)

        out_tokens, out_tags = [], []
        i = 0
        while i < len(tokens):
            tok, tag = tokens[i], tags[i]
            if tag == "B-" + mate_tag:
                out_tokens.extend(repl_tokens)
                out_tags.extend(repl_tags)
                i += 1
                while (i < len(tokens)) and (tokens[i] == "I-" + mate_tag):
                    i += 1
            else:
                out_tokens.append(tok)
                out_tags.append(tag)
                i += 1
        return out_tokens, out_tags

    @staticmethod
    def remove_mate_gooser_name(
        tokens: List[str], tags: List[str], mate_tag: str = "MATE-GOOSER"
    ) -> Tuple[List[str], List[str]]:
        assert len(tokens) == len(tags), f"tokens({tokens}) and tags({tags}) should have the same length"
        # TODO: uppercase first letter if name was removed
        if "B-" + mate_tag not in tags:
            return tokens, tags
        out_tokens, out_tags = [], []
        i = 0
        while i < len(tokens):
            tok, tag = tokens[i], tags[i]
            if i + 1 < len(tokens):
                if (tok == ",") and (tags[i + 1] == "B-" + mate_tag):
                    # it will be mate gooser name next, skip comma
                    i += 1
                    continue
            if i > 0:
                if (tok == ",") and (tags[i - 1][2:] == mate_tag):
                    # that was mate gooser name, skip comma
                    i += 1
                    continue
            if tag[2:] != mate_tag:
                out_tokens.append(tok)
                out_tags.append(tag)
            i += 1
        return out_tokens, out_tags


LIST_LIST_STR_BATCH = List[List[List[str]]]


class HistoryPersonNormalize:
    """
    Takes batch of dialog histories and normalizes only bot responses.

    Detects mentions of mate user's name and either
    (0) converts them to user's name taken from state
    (1) either removes them.

    Parameters:
        per_tag: tag name that corresponds to a person entity
    """

    def __init__(self, per_tag: str = "PER", **kwargs):
        self.per_normalizer = PersonNormalizer(per_tag=per_tag)

    def __call__(
        self, history_tokens: LIST_LIST_STR_BATCH, tags: LIST_LIST_STR_BATCH, states: List[Dict]
    ) -> Tuple[LIST_LIST_STR_BATCH, LIST_LIST_STR_BATCH]:
        out_tokens, out_tags = [], []
        states = states if states else [{}] * len(tags)
        for u_state, u_hist_tokens, u_hist_tags in zip(states, history_tokens, tags):
            # TODO: normalize bot response history
            pass
        return out_tokens, out_tags


class MyselfDetector:
    """
    Finds first mention of a name and sets it as a user name.

    Parameters:
        person_tag: tag name that corresponds to a person entity
        state_slot: name of a state slot corresponding to a user's name

    """

    def __init__(self, person_tag: str = "PER", **kwargs):
        self.per_tag = person_tag

    def __call__(self, tokens: List[List[str]], tags: List[List[str]], states: List[dict]) -> List[str]:
        names = []
        for u_state, u_toks, u_tags in zip(states, tokens, tags):
            cur_name = u_state["user"]["profile"]["name"]
            new_name = copy(cur_name)
            if not cur_name:
                name_found = self.find_my_name(u_toks, u_tags, person_tag=self.per_tag)
                if name_found is not None:
                    new_name = name_found
            names.append(new_name)
        return names

    @staticmethod
    def find_my_name(tokens: List[str], tags: List[str], person_tag: str) -> str:
        if "B-" + person_tag not in tags:
            return None
        per_start = tags.index("B-" + person_tag)
        per_excl_end = per_start + 1
        while (per_excl_end < len(tokens)) and (tags[per_excl_end] == "I-" + person_tag):
            per_excl_end += 1
        return " ".join(tokens[per_start:per_excl_end])


class NerWithContextWrapper:
    """
    Tokenizers utterance and history of dialogue and gets entity tags for
    utterance's tokens.

    Parameters:
        ner_model: named entity recognition model
        tokenizer: tokenizer to use

    """

    def __init__(self, ner_model: Callable, tokenizer: Callable, context_delimeter: str = None, **kwargs):
        self.ner_model = ner_model
        self.tokenizer = tokenizer
        self.context_delimeter = context_delimeter

    def __call__(
        self, utterances: List[str], history: List[List[str]] = [[]], prev_utterances: List[str] = []
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if prev_utterances:
            history = history or itertools.repeat([])
            history = [hist + [prev] for prev, hist in zip(prev_utterances, history)]
        history_toks = [
            [tok for toks in self.tokenizer(hist or [""]) for tok in toks + [self.context_delimeter] if tok is not None]
            for hist in history
        ]
        utt_toks = self.tokenizer(utterances)
        texts, ranges = [], []
        for utt, hist in zip(utt_toks, history_toks):
            if self.context_delimeter is not None:
                txt = hist + utt + [self.context_delimeter]
            else:
                txt = hist + utt
            ranges.append((len(hist), len(hist) + len(utt)))
            texts.append(txt)

        _, tags = self.ner_model(texts)
        tags = [t[l:r] for t, (l, r) in zip(tags, ranges)]

        return utt_toks, tags


class DefaultPostprocessor:
    def __init__(self) -> None:
        self.person_normalizer = PersonNormalizer(per_tag="PER")

    def __call__(self, dialogs: Sequence[Dialog]) -> Sequence[str]:
        new_responses = []
        for d in dialogs:
            # get tokens & tags
            response = d["utterances"][-1]
            try:
                ner_annotations = response["annotations"].get("ner", {})
                user_name = d["user"]["profile"]["name"]
                # replace names with user name
                if ner_annotations and (response["active_skill"] == "chitchat"):
                    response_toks_norm, _ = self.person_normalizer(
                        [ner_annotations["tokens"]], [ner_annotations["tags"]], [user_name]
                    )
                    response_toks_norm = response_toks_norm[0]
                    # detokenize
                    new_responses.append(detokenize(response_toks_norm))
                else:
                    new_responses.append(response["text"])
            except KeyError:
                new_responses.append(response["text"])

        return new_responses
