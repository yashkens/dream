import logging
import re
import time
from os import getenv

import requests
import sentry_sdk
from flask import Flask, request, jsonify
from nltk import tokenize

from common.constants import CAN_CONTINUE
from common.universal_templates import if_lets_chat_about_topic
from common.utils import join_sentences_in_or_pattern


sentry_sdk.init(getenv('SENTRY_DSN'))

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

ANNTR_HISTORY_LEN = 3
AA_FACTOR = 0.05
DEFAULT_CONFIDENCE = 0.9
HAS_SPEC_CHAR_CONFIDENCE = 0.85
HIGHEST_CONFIDENCE = 0.99
LETS_CHAT_ABOUT_CONFIDENDENCE = 0.985
NOUNPHRASE_ENTITY_CONFIDENCE = 0.95
KNOWLEDGE_GROUNDING_SERVICE_URL = getenv('KNOWLEDGE_GROUNDING_SERVICE_URL')
special_char_re = re.compile(r'[^0-9a-zA-Z \-\.\?,!]+')
tokenizer = tokenize.RegexpTokenizer(r'\w+')

with open("./google-english-no-swears.txt", "r") as f:
    UNIGRAMS = set(f.read().splitlines())


def get_entities(utt):
    entities = []

    for ent in utt["annotations"].get("ner", []):
        if not ent:
            continue
        ent = ent[0]["text"].lower()
        if ent not in UNIGRAMS and not (ent == "alexa" and utt["text"].lower()[:5] == "alexa"):
            entities.append(ent)
    return entities


def get_annotations_from_dialog(utterances, annotator_name, key_name):
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
        if isinstance(annotation, dict) and key_name in annotation:
            value = annotation.get(key_name, "")

        # include only non-empty strs
        if value:
            result_values.append([(len(utterances) - i - 1) * 0.01, value])
    return result_values


@app.route("/respond", methods=['POST'])
def respond():
    print('response generation started')
    st_time = time.time()
    dialogs_batch = request.json["dialogs"]
    input_batch = []

    nounphrases = []
    entities = []
    annotations_depths = []
    lets_chat_about_flags = []
    for dialog in dialogs_batch:
        try:
            user_input_text = dialog["human_utterances"][-1]["text"]

            nounphrases.append(
                re.compile(join_sentences_in_or_pattern(
                    dialog["human_utterances"][-1].get("annotations", {}).get("cobot_nounphrases", [])
                ), re.IGNORECASE))
            entities.append(
                re.compile(join_sentences_in_or_pattern(
                    get_entities(dialog["human_utterances"][-1])), re.IGNORECASE))
            lets_chat_about_intent = dialog["human_utterances"][-1].get("annotations", {}).get(
                "intent_catcher", {}).get("lets_chat_about", {}).get("detected", False)
            lets_chat_about_flags.append(if_lets_chat_about_topic(user_input_text.lower()) or lets_chat_about_intent)

            user_input_history = [i["text"] for i in dialog["utterances"]]
            user_input_history = '\n'.join(user_input_history)

            user_input_knowledge = ""
            anntrs_knowledge = ""
            # look for kbqa/odqa text in ANNTR_HISTORY_LEN previous human utterances
            annotators = {
                "odqa": "paragraph",
                "kbqa": "answer"
            }
            annotations_depth = {}
            for anntr_name, anntr_key in annotators.items():
                prev_anntr_outputs = get_annotations_from_dialog(
                    dialog["utterances"][-ANNTR_HISTORY_LEN * 2 - 1:],
                    anntr_name,
                    anntr_key
                )
                logger.debug(f"Prev {anntr_name} {anntr_key}s: {prev_anntr_outputs}")
                # add final dot to kbqa answer to make it a sentence
                if prev_anntr_outputs and anntr_name == "kbqa":
                    prev_anntr_outputs[-1][1] += "."
                # concat annotations separated by space to make a paragraph
                if prev_anntr_outputs and prev_anntr_outputs[-1][1] != "Not Found":
                    anntrs_knowledge += prev_anntr_outputs[-1][1] + " "
                    annotations_depth[anntr_name] = prev_anntr_outputs[-1][0]
            if anntrs_knowledge:
                user_input_knowledge += '\n'.join(tokenize.sent_tokenize(anntrs_knowledge))
            user_input_checked_sentence = tokenize.sent_tokenize(
                user_input_knowledge)[0] if user_input_knowledge else ""

            user_input = {
                'checked_sentence': user_input_checked_sentence,
                'knowledge': user_input_knowledge,
                'text': user_input_text,
                'history': user_input_history
            }
            annotations_depths.append(annotations_depth)
            input_batch.append(user_input)

        except Exception as ex:
            sentry_sdk.capture_exception(ex)
            logger.exception(ex)

    try:
        resp = requests.post(KNOWLEDGE_GROUNDING_SERVICE_URL, json={'batch': input_batch}, timeout=1.5)
        responses = resp.json()
        confidences = []
        attributes = []

        for i, dialog in enumerate(dialogs_batch):
            attr = {
                "knowledge_paragraph": input_batch[i]["knowledge"],
                "knowledge_checked_sentence": input_batch[i]["checked_sentence"],
                "can_continue": CAN_CONTINUE
            }
            already_was_active = int(dialog["bot_utterances"][-1].get("active_skill", "")
                                     == "knowledge_grounding_skill") if len(
                dialog["bot_utterances"]) > 0 else 0
            short_long_response = 0.18 * int(len(tokenizer.tokenize(responses[i])) > 20 or len(
                tokenizer.tokenize(responses[i])) < 4)
            if lets_chat_about_flags[i]:
                confidence = LETS_CHAT_ABOUT_CONFIDENDENCE
                - annotations_depths[i].get("odqa", 0.0) - AA_FACTOR * already_was_active - short_long_response
            else:
                confidence = DEFAULT_CONFIDENCE - annotations_depths[i].get("odqa", 0.0)
                - AA_FACTOR * already_was_active - short_long_response

            if nounphrases[i].search(responses[i]) or entities[i].search(responses[i]):
                confidence = NOUNPHRASE_ENTITY_CONFIDENCE
                - annotations_depths[i].get("odqa", 0.0) - AA_FACTOR * already_was_active - short_long_response
            if (nounphrases[i].search(responses[i]) or entities[i].search(responses[i])) and lets_chat_about_flags[i]:
                confidence = HIGHEST_CONFIDENCE - annotations_depths[i].get("odqa", 0.0)
                - AA_FACTOR * already_was_active - short_long_response
            if special_char_re.search(responses[i]):
                confidence = HAS_SPEC_CHAR_CONFIDENCE
                - annotations_depths[i].get("odqa", 0.0) - AA_FACTOR * already_was_active - short_long_response

            attributes.append(attr)
            confidences.append(confidence)

    except Exception as ex:
        sentry_sdk.capture_exception(ex)
        logger.exception(ex)
        responses = [""] * len(dialogs_batch)
        confidences = [0.] * len(dialogs_batch)
        attributes = [{}] * len(dialogs_batch)

    logger.info(f"Respond exec time: {time.time() - st_time}")
    return jsonify(list(zip(responses, confidences, attributes)))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
