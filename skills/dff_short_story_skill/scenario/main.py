import logging

from df_engine.core.keywords import GLOBAL, TRANSITIONS, RESPONSE
from df_engine.core import Actor
import df_engine.conditions as cnd

import common.dff.integration.condition as int_cnd

from . import condition as loc_cnd
from . import response as loc_rsp

logger = logging.getLogger(__name__)

# ("story_flow", "gpt_topic"): cnd.all(
#     [loc_cnd.has_story_intent,
#      cnd.neg(loc_cnd.needs_scripted_story)]
# ),
# ("story_flow", "gpt_keyword_story"): cnd.all(
#     [cnd.neg(loc_cnd.has_story_intent),
#      cnd.neg(loc_cnd.needs_scripted_story)]
# )

flows = {
    GLOBAL: {TRANSITIONS: {
        ("story_flow", "gpt_topic"): cnd.all(
                    [loc_cnd.has_story_intent,
                    cnd.neg(loc_cnd.needs_scripted_story),
                    loc_cnd.should_return]
                ),
        ("story_flow", "gpt_keyword_story"): cnd.all(
                    [cnd.neg(loc_cnd.has_story_intent),
                    cnd.neg(loc_cnd.needs_scripted_story),
                    loc_cnd.should_return]
                ),
        ("story_flow", "fallback_node"): cnd.all(
            [loc_cnd.needs_scripted_story,
             loc_cnd.should_return])}
    },
    "story_flow": {
        "start_node": {
            RESPONSE: "",
            TRANSITIONS: {
                "gpt_keyword_story": cnd.all(
                    [cnd.neg(loc_cnd.has_story_intent),
                    cnd.neg(loc_cnd.needs_scripted_story),
                    loc_cnd.should_return]
                ),
                "gpt_topic": cnd.all(
                    [loc_cnd.has_story_intent,
                    cnd.neg(loc_cnd.needs_scripted_story),
                    loc_cnd.should_return]
                ),
                "choose_story_node": cnd.all(
                    [
                        loc_cnd.is_tell_me_a_story,
                        loc_cnd.has_story_type,
                        loc_cnd.has_story_left,
                        loc_cnd.needs_scripted_story,
                        loc_cnd.should_return
                    ]
                ),
                "which_story_node": cnd.all(
                    [
                        loc_cnd.is_tell_me_a_story,
                        cnd.neg(loc_cnd.has_story_type),
                        loc_cnd.needs_scripted_story,
                        loc_cnd.should_return
                    ]),
            },
        },
        "choose_story_node": {
            RESPONSE: loc_rsp.choose_story,
            TRANSITIONS: {
                "tell_punchline_node": cnd.any([int_cnd.is_yes_vars, int_cnd.is_do_not_know_vars]),
                "which_story_node": int_cnd.is_no_vars,
            },
        },
        "which_story_node": {
            RESPONSE: loc_rsp.which_story,
            TRANSITIONS: {"choose_story_node": cnd.all([loc_cnd.has_story_type, loc_cnd.has_story_left])},
        },
        "tell_punchline_node": {
            RESPONSE: loc_rsp.tell_punchline,
        },
        "fallback_node": {
            RESPONSE: loc_rsp.fallback,
            TRANSITIONS: {
                "which_story_node": cnd.all(
                    [
                        loc_cnd.is_asked_for_a_story,
                        int_cnd.is_yes_vars,
                        loc_cnd.needs_scripted_story
                    ])
                },
        },
        "gpt_topic": {
            RESPONSE: loc_rsp.choose_topic,
            TRANSITIONS: {
                # "gpt_story": cnd.true()
                "gpt_story": loc_cnd.prev_is_question
            }
        },
        "gpt_story": {
            RESPONSE: loc_rsp.generate_prompt_story
        },
        "gpt_keyword_story": {
            RESPONSE: loc_rsp.generate_story,
        }
    },
}

actor = Actor(flows, start_label=("story_flow", "start_node"), fallback_label=("story_flow", "fallback_node"))
