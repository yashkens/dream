import json
import logging
import numpy as np
import random
import re
import time
from copy import deepcopy
from os import getenv

import requests
import sentry_sdk
from flask import Flask, request, jsonify
from nltk import pos_tag, tokenize

from common.constants import CAN_NOT_CONTINUE
from common.universal_templates import if_chat_about_particular_topic, if_choose_topic
from common.utils import get_intents, join_sentences_in_or_pattern, join_words_in_or_pattern, get_topics, get_entities
from common.response_selection import ACTIVE_SKILLS

sentry_sdk.init(getenv("SENTRY_DSN"))

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DEFAULT_ANNTR_HISTORY_LEN = 0
TOP_N_FACTS = 2
AA_FACTOR = 0.05
ABBRS_CONFIDENCE = 0.8
DEFAULT_CONFIDENCE = 0.88
HAS_SPEC_CHAR_CONFIDENCE = 0.0
HIGHEST_CONFIDENCE = 0.99
KG_ACTIVE_DEPTH = 2
LETS_CHAT_ABOUT_CONFIDENDENCE = 0.6
NOUNPHRASE_ENTITY_CONFIDENCE = 0.95
KNOWLEDGE_GROUNDING_SERVICE_URL = getenv("KNOWLEDGE_GROUNDING_SERVICE_URL")
ACTIVE_SKILLS.remove("personal_info_skill")
DFF_SKILLS = deepcopy(ACTIVE_SKILLS)
DFF_ANNTR_HISTORY_LEN = 1

special_char_re = re.compile(r"[^0-9a-zA-Z \-\.\'\?,!]+")
greetings_farewells_re = re.compile(
    join_words_in_or_pattern(
        [
            "have .* day",
            "have .* night",
            ".* bye",
            r"\bbye",
            "goodbye",
            "hello",
            "(it|its|it's|nice|thank you|thanks).* chatting.*",
            "(it|its|it's|nice|thank you|thanks).* talking.*",
            ".* chatting with you.*",
            "hi",
            "good morning",
            "good afternoon",
            "good luck",
            "great chat",
            "get off.*",
            "thanks for the chat",
            "thank.* for .* chat",
        ]
    ),
    re.IGNORECASE,
)
tokenizer = tokenize.RegexpTokenizer(r"\w+")

with open("./google-english-no-swears.txt", "r") as f:
    UNIGRAMS = set(f.read().splitlines())
with open("./abbreviations_acronyms_list.txt", "r") as f:
    ABBRS = re.compile(join_words_in_or_pattern(list(f.read().splitlines())), re.IGNORECASE)
with open("./topics_facts.json") as f:
    TOPICS_FACTS = json.load(f)


def check_dffs(bot_uttrs):
    flag = False
    if len(bot_uttrs) > 1:
        last_utt_skill = bot_uttrs[-1].get("active_skill", "")
        last_but_one_utt_skill = bot_uttrs[-2].get("active_skill", "")
        if last_utt_skill in DFF_SKILLS:
            if last_but_one_utt_skill == last_utt_skill:
                flag = True
    return flag


def get_named_entities(utt):
    entities = []

    for ent in get_entities(utt, only_named=True, with_labels=False):
        if ent not in UNIGRAMS and not (ent == "alexa" and utt["text"].lower()[:5] == "alexa"):
            entities.append(ent)
    return entities


def get_news(uttr, which):
    result_values = []
    annotation = uttr.get("annotations", {}).get("news_api_annotator", [])
    for news_item in annotation:
        if news_item.get("which", "") == which:
            title = news_item.get("news", {}).get("title", "")
            text = news_item.get("news", {}).get("description", "")
            if text:
                result_values.append({"title": title, "description": text})
    return result_values


def get_fact_random(utterances):
    result_values = []
    for i, uttr in enumerate(utterances):
        values = uttr.get("annotations", {}).get("fact_random", {}).get("facts", [])
        if values:
            for v in values:
                value = v.get("fact", "")
                if value:
                    result_values.append([(len(utterances) - i - 1) * 0.01, value])

    return result_values


def get_annotations_from_dialog(utterances, annotator_name, key_name=None):
    """
    Extract list of strings with values of specific key <key_name>
    from annotator <annotator_name> dict from given dialog utterances.

    Args:
        utterances: utterances, the first one is user's reply
        annotator_name: name of target annotator
        key_name: name of target field from annotation dict

    Returns:
        list of strings with values of specific key from specific annotator
    """
    result_values = []
    for i, uttr in enumerate(utterances):
        annotation = uttr.get("annotations", {}).get(annotator_name, {})
        value = ""
        if isinstance(annotation, dict):
            if key_name in annotation:
                # check if odqa has nonempty answer along with a paragraph
                if annotator_name == "kbqa":
                    value = annotation.get(key_name, "")
                # include only non-empty strs
                if value:
                    result_values.append([(len(utterances) - i - 1) * 0.01, value])
            if "facts" in annotation:
                values = deepcopy(annotation["facts"])
                for value in values[:2]:
                    result_values.append([(len(utterances) - i - 1) * 0.01, value])
        if isinstance(annotation, list):
            values = deepcopy(annotation)
            for value in values[:2]:
                result_values.append([(len(utterances) - i - 1) * 0.01, value])

    return result_values


def get_spacy_nounphrases(utt):
    cob_nounphs = get_entities(utt, only_named=False, with_labels=False)
    spacy_nounphrases = []
    for ph in cob_nounphs:
        if not pos_tag([ph])[0][1].startswith("VB"):
            spacy_nounphrases.append(ph)
    return spacy_nounphrases


def get_intents_flags(utt):
    special_intents = [
        "cant_do",
        "repeat",
        "weather_forecast_intent",
        "what_are_you_talking_about",
        "what_can_you_do",
        "what_is_your_job",
        "what_is_your_name",
        "what_time",
        "where_are_you_from",
        "who_made_you",
    ]
    detected_intents = get_intents(utt, which="intent_catcher")
    lets_chat_about_flag = if_chat_about_particular_topic(utt)
    special_intents_flag = any([si in detected_intents for si in special_intents])
    return lets_chat_about_flag, special_intents_flag


def get_lets_chat_topic(lets_chat_about_flag, utt):
    lets_chat_topic = ""
    COBOT_DA_FILE_TOPICS_MATCH = {
        "Entertainment_Movies": "movies",
        "Entertainment_Music": "music",
        "Science_and_Technology": "science",
        "Sports": "sports",
        "Games": "games",
        "Movies_TV": "movies",
        "SciTech": "science",
        "Psychology": "emotions",
        "Music": "music",
        "Food_Drink": "food",
        "Weather_Time": "weather",
        "Entertainment": "activities",
        "Celebrities": "celebrities",
        "Travel_Geo": "travel",
        "Art_Event": "art",
    }
    if lets_chat_about_flag:
        _get_topics = get_topics(utt, which="all")
        for topic in _get_topics:
            if topic in COBOT_DA_FILE_TOPICS_MATCH:
                lets_chat_topic = COBOT_DA_FILE_TOPICS_MATCH[topic]
                if lets_chat_topic not in utt["text"]:
                    lets_chat_topic = ""
    return lets_chat_topic


def get_news_api_fact(bot_uttr, human_uttrs, not_switch_or_lets_chat_flag):
    news_api_fact = ""
    if len(human_uttrs) > 1:
        if (bot_uttr.get("active_skill", "") == "news_api_skill") and not_switch_or_lets_chat_flag:
            prev_human_utt_hypotheses = human_uttrs[-2].get("hypotheses", [])
            news_api_hypothesis = [
                h for h in prev_human_utt_hypotheses if (h.get("skill_name", "") == "news_api_skill")
            ]
            if news_api_hypothesis:
                if news_api_hypothesis[0].get("news_status", "") == "opinion_request":
                    news_api_fact = news_api_hypothesis[0].get("curr_news", {}).get("description", "")
    return news_api_fact


def get_knowledge_from_annotators(annotators, uttrs, anntr_history_len):
    user_input_knowledge = ""
    anntrs_knowledge = ""
    # look for kbqa/odqa text in anntr_history_len previous human utterances
    annotations_depth = {}
    for anntr_name, anntr_key in annotators.items():
        prev_anntr_outputs = get_annotations_from_dialog(uttrs[-anntr_history_len * 2 - 1 :], anntr_name, anntr_key)
        logger.debug(f"Prev {anntr_name} {anntr_key}s: {prev_anntr_outputs}")
        # add final dot to kbqa answer to make it a sentence
        if prev_anntr_outputs and anntr_name == "kbqa":
            prev_anntr_outputs[-1][1] += "."
        # concat annotations separated by space to make a paragraph
        if prev_anntr_outputs and prev_anntr_outputs[-1][1] != "Not Found":
            anntrs_knowledge += prev_anntr_outputs[-1][1] + " "
            annotations_depth[anntr_name] = prev_anntr_outputs[-1][0]
    if anntrs_knowledge:
        user_input_knowledge += "\n".join(tokenize.sent_tokenize(anntrs_knowledge))
    return user_input_knowledge, annotations_depth


def space_join(x):
    return " ".join(x) + " " if x else ""


def get_penalties(bot_uttrs, curr_response):
    already_was_active = 0
    if bot_uttrs:
        for bu in range(1, 1 + min(KG_ACTIVE_DEPTH, len(bot_uttrs))):
            already_was_active += int(bot_uttrs[-bu].get("active_skill", "") == "knowledge_grounding_skill")
    already_was_active *= AA_FACTOR
    resp_tokens_len = len(tokenizer.tokenize(curr_response))
    short_long_response = 0.5 * int(resp_tokens_len > 20 or resp_tokens_len < 4)
    return already_was_active, short_long_response


@app.route("/respond", methods=["POST"])
def respond():
    print("response generation started")
    st_time = time.time()
    dialogs_batch = request.json["dialogs"]
    for d_id, dialog in enumerate(dialogs_batch):
        try:
            user_input_text = dialog["human_utterances"][-1]["text"]
            bot_uttr = dialog["bot_utterances"][-1] if len(dialog["bot_utterances"]) > 0 else {}
        except Exception as ex:
            sentry_sdk.capture_exception(ex)
            logger.exception(ex)

    try:
        attributes = []
        confidences = []
        responses = []
        curr_responses = ['You said:' + user_input_text]
        curr_confidences = [1.0]
        # attributes.append(curr_attributes)
        confidences.append(curr_confidences)
        responses.append(curr_responses)

    except Exception as ex:
        sentry_sdk.capture_exception(ex)
        logger.exception(ex)
        responses = [[""]]
        confidences = [[0.0]]
        # attributes = [[{}]]

    logger.info(f"knowledge_grounding_skill exec time: {time.time() - st_time}")
    return jsonify(list(zip(responses, confidences)))

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)
