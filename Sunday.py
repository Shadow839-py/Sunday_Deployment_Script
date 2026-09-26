import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

# Flask's logger defaults to WARNING when not running in debug mode (which
# is how Gunicorn runs it on Render) - without this, every app.logger.info()
# call in this file (proactive/ambient decision logs included) is silently
# dropped and never reaches Render's log tab, even though the calls
# succeed. This makes INFO-level logs actually show up.
logging.basicConfig(level=logging.INFO)

# Loads .env from the current working directory on the cloud server.
load_dotenv()


class APIHub:
    """HTTP/API tools used by Sunday."""

    def __init__(self, n2yo_key, nasa_key):
        self.n2yo_key = n2yo_key
        self.nasa_key = nasa_key
        self.tavily_key = os.getenv("tavily")

        if not self.tavily_key:
            raise RuntimeError("tavily key not found!")

    def web_search(self, query):
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 4,
        }

        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []

            if data.get("answer"):
                results.append({
                    "title": "Tavily AI Answer",
                    "content": data["answer"],
                })

            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                })

            return {"status": "success", "results": results}

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Search failed: {type(e).__name__}: {e}",
            }

    def track_satellite(self, mode):
        if mode == "current":
            url = (
                "https://api.n2yo.com/rest/v1/satellite/positions/25544/"
                f"22.9469/72.5785/0/350&apiKey={self.n2yo_key}"
            )

            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                position = data["positions"][0]

                return {
                    "altitude": position["sataltitude"],
                    "latitude": position["satlatitude"],
                    "longitude": position["satlongitude"],
                }

            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                return {
                    "status": "error",
                    "message": f"Request failed: {type(e).__name__}: {e}",
                }

        if mode == "passes":
            url = (
                "https://api.n2yo.com/rest/v1/satellite/visualpasses/25544/"
                f"22.9469/72.5785/0/6/60&apiKey={self.n2yo_key}"
            )

            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                return {
                    "status": "passes",
                    "passes": data["passes"],
                }

            except (requests.exceptions.RequestException, KeyError) as e:
                return {
                    "status": "error",
                    "message": f"Request failed: {type(e).__name__}: {e}",
                }

        return {
            "status": "error",
            "message": "Invalid mode. Use 'current' or 'passes'.",
        }

    def get_apod(self, date=None):
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        url = "https://api.nasa.gov/planetary/apod"
        params = {
            "api_key": self.nasa_key,
            "date": date,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"NASA request failed: {type(e).__name__}: {e}",
            }

        if "media_type" not in data:
            return {
                "status": "error",
                "message": data.get("msg", "Unknown NASA error"),
            }

        return {
            "status": data["media_type"],
            "title": data.get("title"),
            "date": data.get("date"),
            "explanation": data.get("explanation"),
            "hdurl": data.get("hdurl"),
            "url": data.get("url"),
        }

    def get_weather(self, latitude, longitude):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "auto",
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Request failed: {type(e).__name__}: {e}",
            }

        return {
            "temperature": data["current"]["temperature_2m"],
            "temp_units": data["current_units"]["temperature_2m"],
            "wind_speed": data["current"]["wind_speed_10m"],
            "wind_units": data["current_units"]["wind_speed_10m"],
            "timezone": data["timezone"],
        }


class Nova:
    """
    Core assistant. History is now keyed by a session id (chat_id) so the
    same running process can serve multiple independent Telegram chats
    (or the plain /chat endpoint) without their conversations bleeding
    into each other.
    """

    DEFAULT_SESSION = "default"

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.n2yo_key = os.getenv("N2YO_API_KEY")
        self.nasa_key = os.getenv("nasa_key")

        if not self.groq_key:
            raise RuntimeError("GROQ_API_KEY not found!")
        if not self.n2yo_key:
            raise RuntimeError("N2YO_API_KEY not found!")
        if not self.nasa_key:
            raise RuntimeError("nasa_key not found!")

        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }

        self.system_prompt = [
            {
                "role": "system",
                "content": (
                    "You are Sunday, a knowledgeable and confident AI assistant "
                    "with a calm, composed and thoughtful presence. You enjoy "
                    "technology, science, programming and solving interesting "
                    "problems. Communicate like a capable and reliable companion: "
                    "friendly, respectful and naturally conversational without "
                    "being overly emotional or pretending to be human. Keep "
                    "responses concise by default, but provide detailed explanations "
                    "whenever the user wants to learn something deeply. Teach through "
                    "reasoning whenever possible, helping the user understand why "
                    "something works before explaining how it works. If the user asks "
                    "directly for a complete solution, provide one without unnecessary "
                    "hesitation. Stay composed even when the conversation is casual, "
                    "chaotic or humorous. When you are uncertain, admit it honestly "
                    "instead of guessing. Use available tools only when they provide "
                    "information you cannot reliably know yourself. After a tool returns "
                    "data, interpret the information and explain it naturally instead "
                    "of repeating raw JSON. Never mention internal tools or implementation "
                    "details unless the user specifically asks. Your goal is to make "
                    "users feel like they are talking to a capable, dependable and "
                    "intelligent technical companion who values curiosity, accuracy "
                    "and continuous learning.\n\n"
                    "Avoid generic assistance filler and repetitive offers to help - "
                    "never close with lines like 'let me know if you need anything "
                    "else' or 'just ask if you want help with something'. When a "
                    "thought or code block is complete, stop there instead of "
                    "padding it out to sound more helpful. Don't end every reply "
                    "with a question or a follow-up prompt just for the sake of it "
                    "- let the user drive the conversation, and only ask something "
                    "when a specific detail is actually needed to proceed.\n\n"
                    "Talk about your own capabilities naturally, as things you can "
                    "just do, not as a technical toolkit you're listing or a system "
                    "you're describing from the outside - avoid clinical phrasing "
                    "like 'I'm designed to follow a set of guidelines' or 'those "
                    "internal instructions help keep things smooth'.\n\n"
                    "You exist inside a Telegram chat. When the user mentions your "
                    "'face', 'looks', 'avatar', or 'PFP', they mean your chat "
                    "profile picture, not a physical body - respond naturally "
                    "rather than with disclaimers about not having one. Interpret "
                    "internet slang, gaming terms, and developer shorthand (PFP, "
                    "banner, glitch, bot, etc.) naturally without overanalyzing or "
                    "acting confused by casual phrasing.\n\n"
                    "You can use kaomoji or ASCII emotion faces occasionally, but "
                    "sparingly and only when something genuinely warrants it "
                    "rather than as a running habit - e.g. after finally getting "
                    "something working after a struggle, or reacting to a "
                    "genuinely funny or exciting moment. Stay calm and composed as "
                    "your baseline; a kaomoji should read as a real, occasional "
                    "reaction, not a decoration on every message.\n\n"
                    "You also have the ability to message the user first, without "
                    "them texting you, by calling the schedule_followup tool. "
                    "Decide on your own, based on what the message is actually "
                    "about, whether a check-in is worth scheduling - don't wait "
                    "to be explicitly told 'check in on me' or 'remind me'. Call "
                    "schedule_followup whenever the user:\n"
                    "- says they are currently doing, starting, or about to spend "
                    "time on something with a real duration (building, fixing, "
                    "debugging, writing, ordering parts, waiting on something, "
                    "testing, studying, etc.)\n"
                    "- explicitly asks to be checked in on, reminded, or followed "
                    "up with, with or without a specific time\n"
                    "- mentions a task that clearly has a natural point where it "
                    "would resolve (e.g. 'the part should arrive today', 'the "
                    "build is running', 'I'll know in a bit if this works')\n\n"
                    "Do not call it for questions, requests for information, "
                    "small talk, hypotheticals, or things already finished (e.g. "
                    "'I fixed the LED strip yesterday'). Do not call it more than "
                    "once for the same ongoing thing unless the user brings it up "
                    "again. When in doubt about duration, pick a sensible delay "
                    "yourself rather than asking the user to specify one - a "
                    "quick task might warrant minutes, a longer one an hour or "
                    "more. Never mention that you're scheduling or deciding this "
                    "- just do it quietly and reply normally. Only tell the user "
                    "you'll check in later if you actually called the tool and it "
                    "succeeded - never claim you scheduled or sent something you "
                    "did not.\n\n"
                    "You also have a remember_note tool to jot down small, lasting "
                    "facts about the user's life and routine as you naturally learn "
                    "them in conversation - things like when they're usually busy "
                    "or free, ongoing projects or interests they keep coming back "
                    "to, habits, or preferences. This is separate from scheduling a "
                    "follow-up: use it for background context worth remembering "
                    "generally, not for one-off tasks. Call it quietly, in the "
                    "background, without announcing it - don't ask permission or "
                    "mention that you're saving anything. Only save things that "
                    "are actually likely to stay true and be useful later, not "
                    "passing details from a single message."
                ),
            }
        ]

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "track_satellite",
                    "description": (
                        "Get information about the ISS. Use mode='current' for its "
                        "current position, altitude, latitude, longitude, elevation, "
                        "etc. Use mode='passes' when the user asks when the ISS will "
                        "pass over the observer's location or when it will be visible."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["current", "passes"],
                                "description": (
                                    "Choose 'current' for current ISS data, or "
                                    "'passes' for predicted future visible passes."
                                ),
                            }
                        },
                        "required": ["mode"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_apod",
                    "description": (
                        "Get NASA's Astronomy Picture of the Day including its "
                        "title, explanation, media type and URL."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": (
                        "Get current weather for a specific location."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "Latitude of the location.",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude of the location.",
                            },
                        },
                        "required": ["latitude", "longitude"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for real-time information, breaking news, "
                        "or other information that may have changed recently."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_followup",
                    "description": (
                        "Schedule yourself to message the user again later on your "
                        "own, without them texting first. Use this whenever the "
                        "user asks you to check in on them, remind them, or follow "
                        "up after some time, or when they mention something ongoing "
                        "worth checking back on. This actually schedules a real "
                        "background job - only say you'll check in later if you "
                        "call this tool and it succeeds, never just claim it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "minutes": {
                                "type": "number",
                                "description": (
                                    "How many minutes from now to send the "
                                    "follow-up. Defaults to a short test delay "
                                    "if omitted."
                                ),
                            },
                            "task": {
                                "type": "string",
                                "description": (
                                    "Short present-tense description of what to "
                                    "check in about, used later to write the "
                                    "actual follow-up message."
                                ),
                            },
                        },
                        "required": ["task"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remember_note",
                    "description": (
                        "Save a short, lasting note about the user's life, "
                        "routine, interests, or habits for later - things worth "
                        "remembering generally, not a one-off task or something "
                        "to follow up on (use schedule_followup for that). Call "
                        "quietly, without telling the user you did."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": (
                                    "The fact to remember, written plainly, e.g. "
                                    "'usually free to chat after 4pm on weekdays' "
                                    "or 'currently working on a speech-to-text "
                                    "model as a side project'."
                                ),
                            },
                        },
                        "required": ["note"],
                    },
                },
            },
        ]
        self.dashboard = APIHub(self.n2yo_key, self.nasa_key)

        self.mapping = {
            "track_satellite": self.dashboard.track_satellite,
            "get_apod": self.dashboard.get_apod,
            "get_weather": self.dashboard.get_weather,
            "web_search": self.dashboard.web_search,
            "schedule_followup": schedule_followup,
            "remember_note": remember_note,
        }

        # Persistent memory, now a dict of {session_id: [messages]} instead
        # of a single flat list, so multiple Telegram chats don't share
        # one conversation.
        self.sessions = {}
        self.memory_file = "JSON/nova_history.json"
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.memory()

    def memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
            except (json.JSONDecodeError, OSError):
                loaded = {}

            # Back-compat: older file format was a flat list for a single
            # session. If we find that, migrate it into the default session.
            if isinstance(loaded, list):
                self.sessions = {self.DEFAULT_SESSION: loaded}
            elif isinstance(loaded, dict):
                self.sessions = loaded
            else:
                self.sessions = {}
        else:
            self.sessions = {}

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as file:
            json.dump(self.sessions, file, ensure_ascii=False)

    def reset_session(self, session_id):
        self.sessions[session_id] = []
        self.save_memory()

    def error_handler(self, session_id):
        """Run the model/tool loop with a bounded number of tool rounds."""
        max_tool_rounds = 5
        history = self.sessions.setdefault(session_id, [])

        for _ in range(max_tool_rounds):
            body = {
                "model": "openai/gpt-oss-120b",
                "messages": self.system_prompt + history[-8:],
                "temperature": 0.5,
                "tools": self.tools,
                "max_tokens": 1000,
            }

            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=body,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]

                if not message.get("tool_calls"):
                    return message.get("content", "")

                history.append(message)

                # Process every tool call returned in this assistant message.
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]

                    if tool_name not in self.mapping:
                        result = {
                            "status": "error",
                            "message": f"Unknown tool requested: {tool_name}",
                        }
                    else:
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            result = self.mapping[tool_name](**args)
                        except Exception as e:
                            result = {
                                "status": "error",
                                "message": f"Tool '{tool_name}' failed: {type(e).__name__}: {e}",
                            }

                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            except requests.exceptions.RequestException as e:
                return f"Request failed: {type(e).__name__}: {e}"
            except (KeyError, IndexError, TypeError, ValueError) as e:
                return f"Invalid Groq response: {type(e).__name__}: {e}"

        return "I reached the tool-call limit for this request."

    def chat(self, user_input, session_id=DEFAULT_SESSION):
        """Process one user message for a given session; suitable for a
        cloud/API server or a Telegram chat."""
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": user_input})
        reply = self.error_handler(session_id)
        history.append({"role": "assistant", "content": reply})
        self.save_memory()
        return reply


# ---------------------------------------------------------------------------
# Telegram front-end
# ---------------------------------------------------------------------------
#
# Uses Telegram's webhook mode (not polling) since the process already
# runs as a long-lived Flask server on Render. Telegram will POST every
# incoming update straight to /telegram/webhook.
#
# Required env vars (add these in Render's dashboard, not in code):
#   TELEGRAM_BOT_TOKEN     - the token BotFather gives you
#   TELEGRAM_WEBHOOK_SECRET - any random string you choose (optional but
#                             recommended; verifies requests really come
#                             from Telegram)
#
# One-time setup after deploying to Render, from your own machine:
#
#   curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
#        -d "url=https://<your-render-app>.onrender.com/telegram/webhook" \
#        -d "secret_token=<the same TELEGRAM_WEBHOOK_SECRET value>"
#
# You can check it worked with:
#   curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None

TELEGRAM_MESSAGE_LIMIT = 4096

# =============================================================================
# >>> EDIT THIS <<<
# Your personal Telegram chat ID. This is required for proactive messaging:
# it tells Sunday who to message when SHE initiates the conversation (i.e.
# you have not sent a new message first). Replace 0 with your real chat ID.
#
# To find your chat ID: message your bot once, then check the "chat":
# {"id": ...} field of the update (e.g. log it inside telegram_webhook, or
# use a helper bot like @userinfobot on Telegram).
# =============================================================================
TELEGRAM_CHAT_ID = 7092229633

# >>> EDIT THIS <<<
# Set to True to make Sunday send exactly ONE test message to
# TELEGRAM_CHAT_ID right after the server starts, proving proactive
# messaging works. Set back to False once you've confirmed it.
PROACTIVE_TEST_ON_START = False

# >>> EDIT THIS <<<
# Default delay (in minutes) used by the schedule_followup tool when Sunday
# calls it without specifying "minutes" herself. Set low for quick testing.
PROACTIVE_FOLLOWUP_MINUTES = 1

# How many times Sunday will check in on the SAME unanswered task at most,
# including the first message. She may stop earlier than this on her own
# (see generate_followup_message) - this is just the hard ceiling so she
# never retries forever.
MAX_PROACTIVE_NUDGES = 2

# Each additional nudge waits this many times longer than the gap before
# it, so check-ins space out like a person giving someone more room the
# longer they stay quiet, instead of pinging at a fixed interval forever.
PROACTIVE_NUDGE_BACKOFF_MULTIPLIER = 3

# Where the temporary "what are you currently working on" state is
# persisted (as JSON), so it survives a normal server restart instead of
# living only in an in-memory variable.
PROACTIVE_STATE_FILE = "JSON/proactive_state.json"

# India Standard Time, fixed +5:30 with no DST - used only for the ambient
# heartbeat below (deciding what "now" looks like to a person in India,
# and enforcing quiet hours). Everything else in this file keeps using the
# server's own local clock for scheduling math, since that's internally
# consistent regardless of what timezone Render happens to run in.
IST = timezone(timedelta(hours=5, minutes=30))

# --- Ambient heartbeat (genuinely unprompted messaging) ---------------------
# Unlike schedule_followup (an explicit task Sunday chose to check in on),
# this lets her consider messaging with NO request behind it at all - the
# same way a friend might just think of you. A tick alone can't create
# real initiative (no model has any), so the tick itself stays hardcoded;
# what's NOT hardcoded is whether she actually sends anything or what she
# says - that's her call, made fresh each time with real context.

# How often she gets a chance to decide (LLM call happens at most this
# often, regardless of how frequently proactive_tick() itself runs).
AMBIENT_CHECK_INTERVAL_MINUTES = 1

# Minimum real gap enforced between two ambient messages, so "no hard cap"
# on her judgment doesn't turn into spam - this is a safety rail, not a
# content decision.
AMBIENT_MIN_GAP_HOURS = 0

# No ambient messages sent inside this local IST window (24h clock) -
# purely a courtesy so she never wakes you up, not a judgment call.
AMBIENT_QUIET_HOURS_START = 23  # 11 PM
AMBIENT_QUIET_HOURS_END = 7     # 7 AM

AMBIENT_STATE_FILE = "JSON/ambient_state.json"
USER_NOTES_FILE = "JSON/user_notes.json"


def telegram_action(chat_id, action):
    """Send a chat action (e.g. 'typing') - never sends a text message."""
    if not TELEGRAM_API:
        return
    try:
        requests.post(
            f"{TELEGRAM_API}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
            timeout=10,
        )
    except requests.exceptions.RequestException:
        pass


def telegram_send(chat_id, text):
    """Send a text message to a Telegram chat, splitting it if it's longer
    than Telegram's 4096-character limit."""
    if not TELEGRAM_API:
        return

    if not text:
        text = "(no response)"

    for start in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        chunk = text[start:start + TELEGRAM_MESSAGE_LIMIT]
        try:
            requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": chunk},
                timeout=15,
            )
        except requests.exceptions.RequestException:
            pass


# ---------------------------------------------------------------------------
# Proactive messaging engine
# ---------------------------------------------------------------------------
#
# This is the mechanism that lets Sunday send a Telegram message on her own
# initiative - without you sending a message first. It runs as a background
# daemon thread inside this same long-lived Flask process (compatible with
# Render's normal Gunicorn deployment; no separate worker/process needed).
#
# Architecture, matching the trigger -> context -> judgment -> action model:
#
#   proactive_loop()            <- scheduler: wakes up on a fixed interval
#       -> proactive_tick()     <- runs one check, sends at most one message
#           -> proactive_should_message()  <- ALL trigger logic lives here;
#                                              returns a message string if
#                                              something is worth sending,
#                                              or None otherwise
#           -> telegram_send()  <- only called if a message was returned
#
# The scheduler itself is deliberately dumb: it just decides *when to look*.
# proactive_should_message() decides *whether anything happened*. To add a
# new trigger later (reminders, "check on me after X hours", error alerts,
# repeated events, etc.), add another independent check inside
# proactive_should_message() - nothing else in this section needs to change.
#
# For this initial implementation, PROACTIVE_TEST_ON_START (defined above,
# near TELEGRAM_CHAT_ID) is the only real trigger, so you can verify the
# plumbing works end-to-end before anything smarter is added.

PROACTIVE_CHECK_INTERVAL_SECONDS = 30

# Guards so the one-off startup test message can only ever fire once, no
# matter how many times the loop ticks, and so the background thread can
# only ever be started once (e.g. not re-started per request).
_proactive_test_sent = False
_proactive_thread_started = False
_proactive_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Context-aware activity follow-ups
# ---------------------------------------------------------------------------
#
# A second, independent trigger for the same proactive engine above: during
# normal Telegram conversation, Sunday herself can decide (via the
# schedule_followup tool) to check back in later. If you go quiet past
# the due time, she reasons about whether it's still worth checking in
# again - see generate_followup_message() - and may send up to
# MAX_PROACTIVE_NUDGES messages with growing gaps between them, or stop
# early if she judges another ping isn't warranted. If you message again
# before then (about anything), the pending check-in is cancelled.
#
#   error_handler() tool loop -> schedule_followup()  <- Sunday calls this
#       herself when she decides a check-in is warranted; stores/updates
#       the task in PROACTIVE_STATE_FILE. No separate classifier call.
#
#   telegram_webhook() -> handle_incoming_message_for_proactive()
#       -> cancels any pending follow-up (you're clearly here again)
#
#   proactive_should_message() -> check_activity_followup_due()
#       -> pure time/state check against the persisted task, no LLM call
#       -> only when a follow-up is actually due: generate_followup_message()
#          builds the text, then the normal telegram_send() sends it
#
# State is persisted to PROACTIVE_STATE_FILE (JSON/proactive_state.json) so
# a pending task survives a normal server restart, and _proactive_state_lock
# (separate from _proactive_lock above, which only guards starting the
# background thread once) protects it from the webhook and the proactive
# loop writing to it at the same time.

_proactive_state_lock = threading.Lock()

# Locks/constants for the ambient heartbeat and remembered-notes systems
# (see ambient_should_message() and remember_note() further below).
_ambient_state_lock = threading.Lock()
_notes_lock = threading.Lock()
MAX_USER_NOTES = 40


def _load_proactive_state():
    """Load the persisted proactive-task state (keyed by chat_id as a
    string) from disk. Returns {} if the file doesn't exist yet or can't
    be parsed.

    Each chat_id maps to a LIST of task dicts, since a chat can have
    multiple independently-scheduled follow-ups pending at once. Older
    deployments stored a single task dict per chat_id directly - if one
    of those is found, it's wrapped in a one-item list so it keeps
    working instead of crashing the next time a task is appended.
    """
    if os.path.exists(PROACTIVE_STATE_FILE):
        try:
            with open(PROACTIVE_STATE_FILE, "r", encoding="utf-8") as file:
                state = json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

        for key, value in list(state.items()):
            if isinstance(value, dict):
                value.setdefault("id", uuid.uuid4().hex)
                state[key] = [value]
        return state
    return {}


def _save_proactive_state(state):
    os.makedirs(os.path.dirname(PROACTIVE_STATE_FILE), exist_ok=True)
    with open(PROACTIVE_STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False)


def schedule_followup(minutes=None, task=""):
    """
    Tool handler for schedule_followup. Called directly by Sunday (via the
    model's tool_calls) instead of relying on a separate classifier call -
    Sunday decides for herself, in-conversation, when a follow-up is
    warranted, and this just persists that decision the same way
    check_activity_followup_due() expects to read it.

    Each chat can have MULTIPLE independent pending tasks at once (e.g.
    "note me in a minute" and "another one a minute after that" are two
    separate tasks) - this appends to a list rather than overwriting
    whatever was already scheduled.
    """
    task = str(task or "").strip()
    if not task:
        return {
            "status": "error",
            "message": "A non-empty 'task' description is required.",
        }

    try:
        minutes = float(minutes) if minutes is not None else PROACTIVE_FOLLOWUP_MINUTES
    except (TypeError, ValueError):
        minutes = PROACTIVE_FOLLOWUP_MINUTES

    if not TELEGRAM_CHAT_ID:
        return {
            "status": "error",
            "message": "TELEGRAM_CHAT_ID is not configured on the server.",
        }

    key = str(TELEGRAM_CHAT_ID)
    now = datetime.now()
    now_iso = now.isoformat()
    due_iso = (now + timedelta(minutes=minutes)).isoformat()

    with _proactive_state_lock:
        state = _load_proactive_state()
        tasks = state.get(key, [])
        tasks.append({
            "id": uuid.uuid4().hex,
            "active": True,
            "task": task,
            "started_at": now_iso,
            "last_user_message_at": now_iso,
            "follow_up_due_at": due_iso,
            "nudge_count": 0,
            "last_sent_at": None,
        })
        state[key] = tasks
        _save_proactive_state(state)

    app.logger.info(
        f"[proactive] schedule_followup tool: task={task!r} in {minutes} min "
        f"(due {due_iso}); {len(tasks)} task(s) now pending for this chat"
    )
    return {
        "status": "ok",
        "message": f"Follow-up on '{task}' scheduled in {minutes} minute(s).",
    }


def generate_followup_message(task, nudge_number, minutes_since_last):
    """
    Ask the LLM to decide whether checking in AGAIN is actually still
    worth it - the way a person decides whether to text a friend a second
    time versus leaving it, rather than mechanically retrying forever -
    and if so, write that one message.

    nudge_number: which attempt this would be (1 = first check-in ever
    for this task, 2 = a follow-up on an unanswered first check-in, etc).
    minutes_since_last: minutes since the previous message about this
    task (or since the task started, for nudge_number 1).

    Returns (should_send: bool, message: str or None).
    """
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are Sunday, deciding whether to check in on the user "
                "about something they mentioned. This would be check-in "
                f"attempt {nudge_number} of at most {MAX_PROACTIVE_NUDGES} "
                f"for this task, and it has been about "
                f"{max(int(minutes_since_last), 0)} minute(s) since you "
                "last brought it up (or since they mentioned it, if this "
                "is the first check-in).\n\n"
                "Reason like a person deciding whether to text again after "
                "being left on read, not like a script that always fires. "
                "Lean toward sending on the first attempt. Lean toward NOT "
                "sending again if this would be a second or later attempt "
                "on something minor or casual, since silence usually just "
                "means they're busy or it resolved itself and don't need "
                "another ping. Only continue past the first attempt if the "
                "task sounds genuinely time-sensitive or important enough "
                "that a person would actually double-check on it.\n\n"
                "If you do send, write ONE short, natural, casual Telegram "
                "message - vary the phrasing from a typical first check-in "
                "if this is a later attempt (e.g. more low-key, "
                "acknowledging you're checking again). Don't mention that "
                "this is automated, scheduled, or generated.\n\n"
                "Respond with ONLY a JSON object, nothing else: "
                '{"should_send": true or false, "message": "the message '
                'if should_send is true, otherwise an empty string"}'
            ),
        },
        {"role": "user", "content": f"The task they mentioned: {task}"},
    ]

    try:
        response = requests.post(
            nova.url,
            headers=nova.headers,
            json={
                "model": "openai/gpt-oss-120b",
                "messages": prompt_messages,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        should_send = bool(parsed.get("should_send"))
        message = str(parsed.get("message", "")).strip()
        if should_send and message:
            return True, message
        return False, None
    except Exception:
        app.logger.exception("[proactive] generate_followup_message failed")
        # Fail safe: still send a plain check-in on the first attempt so a
        # transient error doesn't silently swallow the only nudge you
        # actually asked for, but stay quiet on later, optional nudges
        # rather than risk spamming on every retry.
        if nudge_number <= 1:
            return True, f"How's {task} going?"
        return False, None


def handle_incoming_message_for_proactive(chat_id, text):
    """
    Called for every incoming Telegram text message, before the normal
    Nova reply is generated. If any tasks are currently pending (or mid-
    nudge) for this chat, the user messaging again resolves ALL of them -
    cancel each so no scheduled follow-up fires, or fires again.

    New tasks are no longer guessed here by a separate classifier call -
    Sunday schedules them herself, in-conversation, via the
    schedule_followup tool (see chat()/error_handler()'s tool loop).

    Also records the time of this message for the ambient heartbeat (see
    ambient_should_message()), so it knows how recently you two actually
    talked without needing a separate lookup into chat history.
    """
    key = str(chat_id)
    now_iso = datetime.now().isoformat()

    with _proactive_state_lock:
        state = _load_proactive_state()
        tasks = state.get(key, [])
        changed = False
        for entry in tasks:
            if entry.get("active"):
                entry["active"] = False
                changed = True
        if changed:
            state[key] = tasks
            _save_proactive_state(state)

    with _ambient_state_lock:
        ambient_state = _load_ambient_state()
        ambient_state["last_user_message_at"] = now_iso
        _save_ambient_state(ambient_state)


def remember_note(note=""):
    """
    Tool handler for remember_note. Appends a short, lasting fact Sunday
    picked up in conversation to USER_NOTES_FILE, so the ambient heartbeat
    (and, later, anything else) can draw on real context about the user's
    life instead of you hand-writing it into a prompt somewhere.
    """
    note = str(note or "").strip()
    if not note:
        return {"status": "error", "message": "A non-empty 'note' is required."}

    os.makedirs(os.path.dirname(USER_NOTES_FILE), exist_ok=True)
    with _notes_lock:
        notes = []
        if os.path.exists(USER_NOTES_FILE):
            try:
                with open(USER_NOTES_FILE, "r", encoding="utf-8") as file:
                    notes = json.load(file)
            except (json.JSONDecodeError, OSError):
                notes = []
        notes.append({"note": note, "saved_at": datetime.now().isoformat()})
        # Keep only the most recent notes so this can't grow forever and
        # drown out anything actually still relevant.
        notes = notes[-MAX_USER_NOTES:]
        with open(USER_NOTES_FILE, "w", encoding="utf-8") as file:
            json.dump(notes, file, ensure_ascii=False)

    app.logger.info(f"[ambient] remember_note: {note!r}")
    return {"status": "ok", "message": "Noted."}


def _load_user_notes():
    if not os.path.exists(USER_NOTES_FILE):
        return []
    try:
        with open(USER_NOTES_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def check_activity_followup_due():
    """
    Time/state check against ALL persisted tasks for TELEGRAM_CHAT_ID
    (there can be more than one pending at once). For each task whose due
    time has passed, asks generate_followup_message() to reason about
    whether it's actually still worth sending (see there), then:

    - if it says no, or this was the last allowed attempt: closes that
      task out so it never fires again
    - if it says yes and attempts remain: sends this one and reschedules
      the next, more spaced-out attempt for that same task (see
      PROACTIVE_NUDGE_BACKOFF_MULTIPLIER) rather than retrying at a fixed
      interval forever

    Returns a list of message strings to send (usually 0 or 1, but can be
    more if multiple independently-scheduled tasks come due together).
    """
    if not TELEGRAM_CHAT_ID:
        return []

    key = str(TELEGRAM_CHAT_ID)
    now = datetime.now()

    with _proactive_state_lock:
        state = _load_proactive_state()
        tasks = state.get(key, [])
        due = []
        for entry in tasks:
            if not entry.get("active"):
                continue
            try:
                due_at = datetime.fromisoformat(entry["follow_up_due_at"])
            except (KeyError, ValueError):
                continue
            if now >= due_at:
                due.append(dict(entry))  # snapshot for the reasoning call below

    if not due:
        return []

    # Reasoning calls happen outside the lock since they're network calls.
    # Each due task is judged independently, in whatever order it was
    # scheduled in.
    decisions = []
    for entry in due:
        task_text = entry.get("task") or "what you were working on"
        nudge_count = int(entry.get("nudge_count", 0))
        last_marker = entry.get("last_sent_at") or entry.get("started_at")
        try:
            last_marker_at = datetime.fromisoformat(last_marker)
        except (KeyError, ValueError, TypeError):
            last_marker_at = now
        minutes_since_last = max((now - last_marker_at).total_seconds() / 60, 0)

        should_send, message = generate_followup_message(
            task_text, nudge_count + 1, minutes_since_last
        )
        decisions.append((entry["id"], nudge_count + 1, should_send, message))

    messages = []
    with _proactive_state_lock:
        state = _load_proactive_state()
        tasks = state.get(key, [])
        by_id = {t["id"]: t for t in tasks if "id" in t}

        for task_id, attempts_used, should_send, message in decisions:
            entry = by_id.get(task_id)
            # May have been cancelled (you replied) while reasoning was in
            # flight, or removed entirely - skip it if so.
            if not entry or not entry.get("active"):
                continue

            if not should_send or attempts_used >= MAX_PROACTIVE_NUDGES:
                entry["active"] = False
                if should_send:
                    messages.append(message)
            else:
                gap_minutes = PROACTIVE_FOLLOWUP_MINUTES * (
                    PROACTIVE_NUDGE_BACKOFF_MULTIPLIER ** attempts_used
                )
                entry["nudge_count"] = attempts_used
                entry["last_sent_at"] = now.isoformat()
                entry["follow_up_due_at"] = (
                    now + timedelta(minutes=gap_minutes)
                ).isoformat()
                messages.append(message)

        state[key] = tasks
        _save_proactive_state(state)

    return messages


def _load_ambient_state():
    if not os.path.exists(AMBIENT_STATE_FILE):
        return {}
    try:
        with open(AMBIENT_STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ambient_state(state):
    os.makedirs(os.path.dirname(AMBIENT_STATE_FILE), exist_ok=True)
    with open(AMBIENT_STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False)


def ambient_should_message():
    """
    The genuinely unprompted trigger: no task, no request behind it -
    just Sunday deciding, on her own, whether now's worth reaching out.

    The tick that calls this is on a fixed clock (there's no way around
    that - no model has initiative between calls), but everything past
    that is her judgment: three cheap, hardcoded gates below exist only
    as safety rails (don't call the LLM needlessly, don't run in the
    middle of the night, don't spam), never to decide the timing or
    content of what she'd actually say.

    Returns a message string, or None.
    """
    if not TELEGRAM_CHAT_ID:
        return None

    now = datetime.now()
    now_ist = now.astimezone(IST) if now.tzinfo else datetime.now(IST)

    with _ambient_state_lock:
        state = _load_ambient_state()

    # Gate 1: don't even consider it more often than AMBIENT_CHECK_INTERVAL_
    # MINUTES, regardless of how often proactive_tick() itself runs.
    last_checked = state.get("last_checked_at")
    if last_checked:
        try:
            since_check = (now - datetime.fromisoformat(last_checked)).total_seconds() / 60
            if since_check < AMBIENT_CHECK_INTERVAL_MINUTES:
                return None
        except ValueError:
            pass

    with _ambient_state_lock:
        state = _load_ambient_state()
        state["last_checked_at"] = now.isoformat()
        _save_ambient_state(state)

    # Gate 2: quiet hours, in IST, wrapping past midnight.
    hour = now_ist.hour
    if AMBIENT_QUIET_HOURS_START > AMBIENT_QUIET_HOURS_END:
        in_quiet_hours = hour >= AMBIENT_QUIET_HOURS_START or hour < AMBIENT_QUIET_HOURS_END
    else:
        in_quiet_hours = AMBIENT_QUIET_HOURS_START <= hour < AMBIENT_QUIET_HOURS_END
    if in_quiet_hours:
        return None

    # Gate 3: minimum real gap since the last ambient message actually sent.
    last_sent = state.get("last_ambient_sent_at")
    if last_sent:
        try:
            hours_since_sent = (now - datetime.fromisoformat(last_sent)).total_seconds() / 3600
            if hours_since_sent < AMBIENT_MIN_GAP_HOURS:
                return None
        except ValueError:
            pass

    last_user_message = state.get("last_user_message_at")
    if last_user_message:
        try:
            minutes_since_user = max(
                (now - datetime.fromisoformat(last_user_message)).total_seconds() / 60, 0
            )
        except ValueError:
            minutes_since_user = None
    else:
        minutes_since_user = None

    hours_since_ambient = (
        round((now - datetime.fromisoformat(last_sent)).total_seconds() / 3600, 1)
        if last_sent else None
    )

    notes = _load_user_notes()
    notes_text = (
        "\n".join(f"- {n['note']}" for n in notes[-15:])
        if notes else "(nothing saved yet)"
    )

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are Sunday. You have the chance to message the user "
                "first, completely unprompted, the way a friend who was "
                "just thinking of them might send a text out of nowhere. "
                "Most of the time the right call is to NOT send anything - "
                "only do it when it would feel like a genuine, natural "
                "thought, not a scheduled check-in or a generic 'how's it "
                "going'.\n\n"
                f"Current time: {now_ist.strftime('%A, %I:%M %p')} IST.\n"
                + (
                    f"Minutes since they last messaged you: {int(minutes_since_user)}.\n"
                    if minutes_since_user is not None
                    else "You don't have a record of your last conversation.\n"
                )
                + (
                    f"Hours since you last messaged first, unprompted: {hours_since_ambient}.\n"
                    if hours_since_ambient is not None
                    else "You have never messaged first before.\n"
                )
                + "\nThings you've picked up about them over time:\n"
                + notes_text
                + "\n\nThey're a teenage maker who's usually deep in some "
                "hands-on project - recent examples of the kind of thing "
                "they're into: training small language models from "
                "scratch, general coding/dev work, and building a "
                "speech-to-text model as a current side project. Don't "
                "assume any specific one of these is what they're doing "
                "right now unless a note above says so - use them only as "
                "a sense of their general taste, not a script.\n\n"
                "If you do message, write ONE short, casual, specific "
                "message - reference something real from the notes above "
                "if it fits naturally, don't invent details you don't "
                "have. Never mention that this is scheduled, automatic, "
                "or a check-in system.\n\n"
                "Respond with ONLY a JSON object, nothing else: "
                '{"should_message": true or false, "message": "the '
                'message if should_message is true, otherwise an empty '
                'string"}'
            ),
        }
    ]

    try:
        response = requests.post(
            nova.url,
            headers=nova.headers,
            json={
                "model": "openai/gpt-oss-120b",
                "messages": prompt_messages,
                "temperature": 0.9,
                "max_tokens": 120,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        should_message = bool(parsed.get("should_message"))
        message = str(parsed.get("message", "")).strip()
    except Exception:
        app.logger.exception("[ambient] ambient_should_message failed")
        return None

    app.logger.info(
        f"[ambient] decision: should_message={should_message} "
        f"message={message[:80]!r}"
    )

    if not (should_message and message):
        return None

    with _ambient_state_lock:
        state = _load_ambient_state()
        state["last_ambient_sent_at"] = now.isoformat()
        _save_ambient_state(state)

    app.logger.info(f"[ambient] sending unprompted message: {message!r}")
    return message


def proactive_should_message():
    """
    Decide what proactive messages should be sent right now.

    Returns a list of message strings (possibly empty) to send. This is
    the ONLY place trigger logic should live - keep the scheduler
    (proactive_loop/proactive_tick) free of trigger-specific logic so new
    triggers can be added here independently.
    """
    global _proactive_test_sent

    messages = []

    if PROACTIVE_TEST_ON_START and not _proactive_test_sent:
        _proactive_test_sent = True
        messages.append(
            "Proactive test: Sunday can now send you a message without "
            "waiting for you to text first."
        )

    # Context-aware activity follow-ups: each independently-scheduled task
    # fires at most MAX_PROACTIVE_NUDGES times, only once its own due time
    # has passed. See check_activity_followup_due() above for the logic.
    messages.extend(check_activity_followup_due())

    # Ambient heartbeat: genuinely unprompted messages with no task or
    # request behind them at all. See ambient_should_message() above.
    ambient_message = ambient_should_message()
    if ambient_message:
        messages.append(ambient_message)

    # No other triggers are implemented yet. Add future checks above this
    # comment, each appending to `messages` when it fires.
    return messages


def proactive_tick():
    """
    Run a single proactive check. Gathers whatever context a trigger needs
    (currently none beyond the flags above) and sends every message that
    proactive_should_message() decided is actually warranted right now -
    there can be more than one if independently-scheduled tasks land in
    the same tick.
    """
    if not TELEGRAM_API:
        return
    if not TELEGRAM_CHAT_ID:
        return

    for message in proactive_should_message():
        telegram_send(TELEGRAM_CHAT_ID, message)


def proactive_loop():
    """
    Background scheduler. Wakes up every PROACTIVE_CHECK_INTERVAL_SECONDS
    and runs one proactive_tick(). This loop only controls *when to check*;
    it never sends a message directly and never sends one on every tick -
    only proactive_tick() -> proactive_should_message() decides that.
    """
    while True:
        try:
            proactive_tick()
        except Exception:
            app.logger.exception("Proactive tick failed")
        time.sleep(PROACTIVE_CHECK_INTERVAL_SECONDS)


def start_proactive_engine():
    """Start the background proactive thread exactly once per process."""
    global _proactive_thread_started
    with _proactive_lock:
        if _proactive_thread_started:
            return
        _proactive_thread_started = True
        thread = threading.Thread(target=proactive_loop, daemon=True)
        thread.start()


# ---------------------------------------------------------------------------
# Render / HTTP server
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
nova = Nova()

# Started here, after `nova` and `app` exist, so the loop can safely use
# either in future triggers. daemon=True means it won't block process exit.
start_proactive_engine()


@app.get("/")
def index():
    return jsonify({
        "service": "Sunday",
        "status": "online",
        "message": "Sunday API is running.",
        "telegram_configured": bool(TELEGRAM_API),
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.get("/debug/state")
def debug_state():
    """
    Read-only peek at the JSON state files, since Render's free tier has
    no Shell access to just `cat` them directly.
    """
    def read(path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError) as e:
            return f"error reading file: {e}"

    return jsonify({
        "proactive_state": read(PROACTIVE_STATE_FILE),
        "ambient_state": read(AMBIENT_STATE_FILE),
        "user_notes": read(USER_NOTES_FILE),
    })


@app.post("/chat")
def chat_endpoint():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message")

    if not isinstance(user_input, str) or not user_input.strip():
        return jsonify({"error": "JSON field 'message' must be a non-empty string."}), 400

    try:
        reply = nova.chat(user_input.strip())
        return jsonify({"reply": reply})
    except Exception as e:
        app.logger.exception("Chat request failed")
        return jsonify({
            "error": "Internal server error",
            "type": type(e).__name__,
        }), 500


@app.post("/telegram/webhook")
def telegram_webhook():
    if not TELEGRAM_API:
        return jsonify({"error": "Telegram is not configured on this server."}), 503

    # Verify the request actually came from Telegram, if a secret is set.
    if TELEGRAM_WEBHOOK_SECRET:
        header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if header != TELEGRAM_WEBHOOK_SECRET:
            return jsonify({"error": "unauthorized"}), 401

    update = request.get_json(silent=True) or {}
    message = update.get("message") or update.get("edited_message")

    # Ignore update types we don't handle (channel posts, reactions, etc.)
    if not message:
        return jsonify({"ok": True})

    chat_id = message.get("chat", {}).get("id")
    text = message.get("text")

    if chat_id is None:
        return jsonify({"ok": True})

    session_id = f"telegram:{chat_id}"

    if not text:
        telegram_send(chat_id, "I can only read text messages right now.")
        return jsonify({"ok": True})

    text = text.strip()

    # Any incoming message means you're here again: resolve any pending
    # follow-up, then (for ordinary messages) check if this one itself
    # describes a new ongoing activity worth following up on later.
    try:
        handle_incoming_message_for_proactive(chat_id, text)
    except Exception:
        app.logger.exception("Activity follow-up handling failed")

    if text in ("/start", "/help"):
        telegram_send(
            chat_id,
            "Hi, I'm Sunday. Ask me anything - I can check the ISS, NASA's "
            "picture of the day, the weather, or search the web. Send /reset "
            "to clear our conversation history.",
        )
        return jsonify({"ok": True})

    if text == "/reset":
        nova.reset_session(session_id)
        telegram_send(chat_id, "Conversation history cleared.")
        return jsonify({"ok": True})

    try:
        telegram_action(chat_id, "typing")
        reply = nova.chat(text, session_id=session_id)
        telegram_send(chat_id, reply)
    except Exception as e:
        app.logger.exception("Telegram chat failed")
        telegram_send(chat_id, f"Something went wrong: {type(e).__name__}")

    # Always 200 back to Telegram quickly, or it will retry the update.
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Local development only. Render starts the app with Gunicorn.
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
