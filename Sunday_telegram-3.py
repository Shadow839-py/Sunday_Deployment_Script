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
                    "and continuous learning."
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
        ]
        self.dashboard = APIHub(self.n2yo_key, self.nasa_key)

        self.mapping = {
            "track_satellite": self.dashboard.track_satellite,
            "get_apod": self.dashboard.get_apod,
            "get_weather": self.dashboard.get_weather,
            "web_search": self.dashboard.web_search,
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
TELEGRAM_CHAT_ID = 0

# >>> EDIT THIS <<<
# Set to True to make Sunday send exactly ONE test message to
# TELEGRAM_CHAT_ID right after the server starts, proving proactive
# messaging works. Set back to False once you've confirmed it.
PROACTIVE_TEST_ON_START = False

# >>> EDIT THIS <<<
# How long Sunday waits, after you mention an ongoing activity, before
# checking in if you haven't messaged again. Change this one number to
# make follow-ups fire sooner or later - nothing else needs to change.
PROACTIVE_FOLLOWUP_MINUTES = 15

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

PROACTIVE_CHECK_INTERVAL_SECONDS = 60

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
# A second, independent trigger for the same proactive engine above: if you
# tell Sunday (in normal Telegram conversation) that you're currently doing
# something, she remembers that as a temporary task and - if you go quiet
# for PROACTIVE_FOLLOWUP_MINUTES - sends ONE natural, contextual check-in.
# If you message again before then (about anything), the pending check-in
# is cancelled.
#
#   telegram_webhook() -> handle_incoming_message_for_proactive()
#       -> cancels any pending follow-up (you're clearly here again)
#       -> detect_ongoing_activity()  <- the ONLY place the LLM is called
#                                         for this feature outside of a due
#                                         follow-up; classifies this one
#                                         message, never runs on a timer
#       -> stores/updates the task in PROACTIVE_STATE_FILE
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


def detect_ongoing_activity(user_message):
    """
    Ask the LLM whether `user_message` says the user is CURRENTLY, ACTIVELY
    doing something - the kind of ongoing task worth checking back in on -
    as opposed to a question, small talk, or something already finished.

    This is a small, separate classification call: it is never added to
    the real conversation history, and it only ever runs while handling an
    incoming message - never on a timer.

    Returns a short present-tense task description string if so, or None
    if the message isn't an activity statement.
    """
    classification_messages = [
        {
            "role": "system",
            "content": (
                "You classify a single chat message to decide whether the "
                "user is telling you they are CURRENTLY, ACTIVELY doing "
                "something right now - the kind of ongoing task where it "
                "would make sense for an assistant to check back in later "
                "and ask how it went.\n\n"
                "Say yes for things like 'I'm working on my AI project', "
                "'I'm debugging the vision model', 'I'm going to spend the "
                "next hour fixing this', 'I'm working on this for a "
                "while'.\n\n"
                "Say no for questions, requests for information, small "
                "talk, or statements about the past (e.g. 'I worked on X "
                "yesterday'). Judge the actual meaning of the message, not "
                "just whether words like 'working' or 'project' appear.\n\n"
                "Respond with ONLY a JSON object and nothing else, in "
                "exactly this shape: "
                '{"is_activity": true or false, "task": "short present-'
                'tense description of the activity if is_activity is '
                'true, otherwise an empty string"}'
            ),
        },
        {"role": "user", "content": user_message},
    ]

    try:
        response = requests.post(
            nova.url,
            headers=nova.headers,
            json={
                "model": "openai/gpt-oss-120b",
                "messages": classification_messages,
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Defensively strip a code fence in case the model wraps its JSON
        # despite being told to return it raw.
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        if parsed.get("is_activity") and str(parsed.get("task", "")).strip():
            return str(parsed["task"]).strip()
    except Exception:
        # Classification is best-effort: on any failure, fail safe by not
        # creating a proactive task rather than guessing.
        pass

    return None


def generate_followup_message(task):
    """Generate a short, natural, contextual follow-up message about
    `task` using the same Groq/Nova model - never a hard-coded template."""
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are Sunday, checking in proactively because some "
                "time has passed since the user mentioned they were doing "
                "something, and they haven't messaged since. Write ONE "
                "short, natural, casual Telegram message checking in on "
                "that specific activity - e.g. asking how it's going or "
                "whether they made progress. One short sentence only. "
                "Don't mention that this is automated, scheduled, or "
                "generated."
            ),
        },
        {"role": "user", "content": f"The user said they were: {task}"},
    ]

    try:
        response = requests.post(
            nova.url,
            headers=nova.headers,
            json={
                "model": "openai/gpt-oss-120b",
                "messages": prompt_messages,
                "temperature": 0.7,
                "max_tokens": 60,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content.strip('"') or f"How's {task} going?"
    except Exception:
        return f"How's {task} going?"


def handle_incoming_message_for_proactive(chat_id, text):
    """
    Called for every incoming Telegram text message, before the normal
    Nova reply is generated.

    1. If a task is currently pending (or was already followed up on) for
       this chat, the user messaging again resolves it - cancel it so the
       scheduled follow-up never fires, or never fires a second time.
    2. For ordinary (non-command) messages, ask the LLM whether this
       message itself describes a new ongoing activity, and if so, store
       a fresh task with a follow-up due PROACTIVE_FOLLOWUP_MINUTES from
       now.
    """
    key = str(chat_id)
    now = datetime.now()

    with _proactive_state_lock:
        state = _load_proactive_state()
        existing = state.get(key)
        if existing and existing.get("active"):
            existing["active"] = False
            state[key] = existing
            _save_proactive_state(state)

    # Slash commands are handled separately and aren't natural-language
    # activity statements - skip the classification call for them.
    if text.startswith("/"):
        return

    task = detect_ongoing_activity(text)
    if not task:
        return

    now_iso = now.isoformat()
    with _proactive_state_lock:
        state = _load_proactive_state()
        state[key] = {
            "active": True,
            "task": task,
            "started_at": now_iso,
            "last_user_message_at": now_iso,
            "follow_up_due_at": (
                now + timedelta(minutes=PROACTIVE_FOLLOWUP_MINUTES)
            ).isoformat(),
            "awaiting_response": False,
        }
        _save_proactive_state(state)


def check_activity_followup_due():
    """
    Pure time/state check (no LLM call) against the persisted task for
    TELEGRAM_CHAT_ID. If a follow-up is due and hasn't already been sent,
    generate it, mark the task as followed-up so it can't fire twice, and
    return the message text. Returns None otherwise.
    """
    if not TELEGRAM_CHAT_ID:
        return None

    key = str(TELEGRAM_CHAT_ID)
    now = datetime.now()
    task = None

    with _proactive_state_lock:
        state = _load_proactive_state()
        entry = state.get(key)

        if not entry or not entry.get("active") or entry.get("awaiting_response"):
            return None

        try:
            due_at = datetime.fromisoformat(entry["follow_up_due_at"])
        except (KeyError, ValueError):
            return None

        if now < due_at:
            return None

        task = entry.get("task") or "what you were working on"
        entry["awaiting_response"] = True
        state[key] = entry
        _save_proactive_state(state)

    # Generated outside the lock since it's a network call.
    return generate_followup_message(task)


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
