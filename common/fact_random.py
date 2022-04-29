import json
import logging
from os import getenv

import requests
import sentry_sdk


sentry_sdk.init(getenv("SENTRY_DSN"))
logger = logging.getLogger(__name__)


FACT_RANDOM_SERVICE_URL = getenv("FACT_RANDOM_SERVICE_URL")
if FACT_RANDOM_SERVICE_URL is None:
    raise RuntimeError("FACT_RANDOM_SERVICE_URL environment variable is not set")
HEADERS = {"Content-Type": "application/json;charset=utf-8"}
TIMEOUT = 0.9


def load_fact_file(path):
    with open(path, "r") as f:
        facts = json.load(f)
    for fact_key in list(facts.keys()):
        facts[fact_key.lower()] = facts.pop(fact_key)
    return facts


def _request_fact_service(entity_substr_list: list, question: str) -> list:
    """Put entities in a batch and send to fact-random service

    Args:
        entity_substr_list: list of entities
        question: question asked by user

    Returns: Flattened list of facts

    """

    request_body = {"text": question, "entities": entity_substr_list}
    try:
        resp = requests.request(
            url=FACT_RANDOM_SERVICE_URL, headers=HEADERS, data=json.dumps(request_body), method="POST", timeout=TIMEOUT
        )
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception(e)
        resp = requests.Response()
        resp.status_code = 504

    if resp.status_code != 200:
        logger.warning(
            f"result status code is not 200: {resp}. result text: {resp.text}; result status: {resp.status_code}"
        )
        response = []
        sentry_sdk.capture_message(
            f"Fact-Random result status code is not 200: {resp}. result text: {resp.text}; "
            f"result status: {resp.status_code}"
        )
    else:
        flat_fact_list = []
        for text_and_fact_data in resp.json():
            facts_list = text_and_fact_data.get("facts")
            facts_for_entity = [f.get("fact") for f in facts_list]
            flat_fact_list += facts_for_entity

        response = flat_fact_list

    return response


def get_fact(entity_substr: str, question: str) -> str:
    """Interface method of fact-random service for a single fact

    Args:
        entity_substr: entity which is used to find facts
        question: string with question in natural language

    Returns: answer as string in natural language

    """

    facts = _request_fact_service([entity_substr], question)
    try:
        fact = facts[0]
    except IndexError:
        fact = ""

    return fact


def get_facts(entity_substr_list: list, question: str) -> list:
    """Interface method of fact-random service for multiple facts

    Args:
        entity_substr_list: list of entities which are used to find facts
        question: string with question in natural language

    Returns: answer as list of strings in natural language

    """

    facts = _request_fact_service(entity_substr_list, question)

    return facts


def get_facts_from_fact_random_annotations(utterance):
    return utterance.get("annotations", {}).get("fact_random", [])


def get_facts_for_entity_from_fact_random_annotations(utterance, entity):
    result = []
    for element in get_facts_from_fact_random_annotations(utterance):
        ent = element.get("entity_substr", "")
        fact = element.get("fact", "")
        if fact and ent == entity:
            result.append(fact)

    return result
