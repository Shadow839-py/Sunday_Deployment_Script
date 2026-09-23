import json
import os
import threading
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

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
                    "did not."
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
        ]
        self.dashboard = APIHub(self.n2yo_key, self.nasa_key)

        self.mapping = {
            "track_satellite": self.dashboard.track_satellite,
            "get_apod": self.dashboard.get_apod,
            "get_weather": self.dashboard.get_weather,
            "web_search": self.dashboard.web_search,
            "schedule_followup": schedule_followup,
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


def _load_proactive_state():
    """Load the persisted proactive-task state dict (keyed by chat_id as a
    string) from disk. Returns {} if the file doesn't exist yet or can't be
    parsed."""
    if os.path.exists(PROACTIVE_STATE_FILE):
        try:
            with open(PROACTIVE_STATE_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}
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

    with _proactive_state_lock:
        state = _load_proactive_state()
        state[key] = {
            "active": True,
            "task": task,
            "started_at": now_iso,
            "last_user_message_at": now_iso,
            "follow_up_due_at": (now + timedelta(minutes=minutes)).isoformat(),
            "nudge_count": 0,
            "last_sent_at": None,
        }
        _save_proactive_state(state)

    app.logger.info(
        f"[proactive] schedule_followup tool: task={task!r} in {minutes} min "
        f"(due {state[key]['follow_up_due_at']})"
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
    Nova reply is generated. If a task is currently pending (or was
    already followed up on) for this chat, the user messaging again
    resolves it - cancel it so the scheduled follow-up never fires, or
    never fires a second time.

    New tasks are no longer guessed here by a separate classifier call -
    Sunday schedules them herself, in-conversation, via the
    schedule_followup tool (see chat()/error_handler()'s tool loop).
    """
    key = str(chat_id)

    with _proactive_state_lock:
        state = _load_proactive_state()
        existing = state.get(key)
        if existing and existing.get("active"):
            existing["active"] = False
            state[key] = existing
            _save_proactive_state(state)


def check_activity_followup_due():
    """
    Time/state check against the persisted task for TELEGRAM_CHAT_ID. If a
    check-in is due, asks generate_followup_message() to reason about
    whether it's actually still worth sending (see there), then:

    - if it says no, or this was the last allowed attempt: closes the
      task out so it never fires again for this activity
    - if it says yes and attempts remain: sends this one and reschedules
      the next, more spaced-out attempt (see PROACTIVE_NUDGE_BACKOFF_
      MULTIPLIER) rather than retrying at a fixed interval forever

    Returns the message text to send, or None.
    """
    if not TELEGRAM_CHAT_ID:
        return None

    key = str(TELEGRAM_CHAT_ID)
    now = datetime.now()

    with _proactive_state_lock:
        state = _load_proactive_state()
        entry = state.get(key)

        if not entry or not entry.get("active"):
            return None

        try:
            due_at = datetime.fromisoformat(entry["follow_up_due_at"])
        except (KeyError, ValueError):
            return None

        if now < due_at:
            return None

        task = entry.get("task") or "what you were working on"
        nudge_count = int(entry.get("nudge_count", 0))
        last_marker = entry.get("last_sent_at") or entry.get("started_at")
        try:
            last_marker_at = datetime.fromisoformat(last_marker)
        except (KeyError, ValueError, TypeError):
            last_marker_at = now
        minutes_since_last = max((now - last_marker_at).total_seconds() / 60, 0)

    # Reasoning call happens outside the lock since it's a network call.
    should_send, message = generate_followup_message(
        task, nudge_count + 1, minutes_since_last
    )

    with _proactive_state_lock:
        state = _load_proactive_state()
        entry = state.get(key)
        # The task may have been cancelled (you replied) while the
        # reasoning call above was in flight - re-check before acting.
        if not entry or not entry.get("active"):
            return None

        attempts_used = nudge_count + 1
        if not should_send or attempts_used >= MAX_PROACTIVE_NUDGES:
            # Either Sunday decided against it, or we've hit the ceiling -
            # close the task out either way so it can't fire again.
            entry["active"] = False
            state[key] = entry
            _save_proactive_state(state)
            return message if should_send else None

        # More attempts remain and Sunday chose to send this one - keep
        # the task open, but back off further before trying again.
        gap_minutes = PROACTIVE_FOLLOWUP_MINUTES * (
            PROACTIVE_NUDGE_BACKOFF_MULTIPLIER ** attempts_used
        )
        entry["nudge_count"] = attempts_used
        entry["last_sent_at"] = now.isoformat()
        entry["follow_up_due_at"] = (
            now + timedelta(minutes=gap_minutes)
        ).isoformat()
        state[key] = entry
        _save_proactive_state(state)

    return message


def proactive_should_message():
    """
    Decide whether a proactive message should be sent right now.

    Returns the message text if something warrants sending one, or None to
    send nothing. This is the ONLY place trigger logic should live - keep
    the scheduler (proactive_loop/proactive_tick) free of trigger-specific
    logic so new triggers can be added here independently.
    """
    global _proactive_test_sent

    if PROACTIVE_TEST_ON_START and not _proactive_test_sent:
        _proactive_test_sent = True
        return (
            "Proactive test: Sunday can now send you a message without "
            "waiting for you to text first."
        )

    # Context-aware activity follow-up: fires at most once per stored task,
    # only when its due time has actually passed. See check_activity_
    # followup_due() above for the full logic.
    followup_message = check_activity_followup_due()
    if followup_message:
        return followup_message

    # No other triggers are implemented yet. Add future checks above this
    # comment, each independently returning a message string when it fires.
    return None


def proactive_tick():
    """
    Run a single proactive check. Gathers whatever context a trigger needs
    (currently none beyond the flags above) and sends at most one message,
    only if proactive_should_message() decided one is actually warranted.
    """
    if not TELEGRAM_API:
        return
    if not TELEGRAM_CHAT_ID:
        return

    message = proactive_should_message()
    if message:
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
