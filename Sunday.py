import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from trading_tools import TradingHub, TRADING_TOOLS_SCHEMA


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

        # Trading tools remain part of the same Groq tool-calling interface.
        self.tools += TRADING_TOOLS_SCHEMA

        self.dashboard = APIHub(self.n2yo_key, self.nasa_key)

        self.mapping = {
            "track_satellite": self.dashboard.track_satellite,
            "get_apod": self.dashboard.get_apod,
            "get_weather": self.dashboard.get_weather,
            "web_search": self.dashboard.web_search,
            "get_crypto_price": self.trading.get_crypto_price,
            "get_technical_indicators": self.trading.get_technical_indicators,
            "get_stock_price": self.trading.get_stock_price,
            "get_fear_greed_index": self.trading.get_fear_greed_index,
            "calculate_position_size": self.trading.calculate_position_size,
            "get_correlation": self.trading.get_correlation,
            "log_trade_journal": self.trading.log_trade_journal,
            "get_trade_journal": self.trading.get_trade_journal,
        }

        # Temporary basic persistent memory.
        self.history = []
        self.memory_file = "JSON/nova_history.json"
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.memory()

    def memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as file:
                    self.history = json.load(file)
            except (json.JSONDecodeError, OSError):
                self.history = []
        else:
            self.history = []

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as file:
            json.dump(self.history, file, ensure_ascii=False)

    def error_handler(self):
        """Run the model/tool loop with a bounded number of tool rounds."""
        max_tool_rounds = 5

        for _ in range(max_tool_rounds):
            body = {
                "model": "openai/gpt-oss-120b",
                "messages": self.system_prompt + self.history[-8:],
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

                self.history.append(message)

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

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            except requests.exceptions.RequestException as e:
                return f"Request failed: {type(e).__name__}: {e}"
            except (KeyError, IndexError, TypeError, ValueError) as e:
                return f"Invalid Groq response: {type(e).__name__}: {e}"

        return "I reached the tool-call limit for this request."

    def chat(self, user_input):
        """Process one user message; suitable for a cloud/API server."""
        self.history.append({"role": "user", "content": user_input})
        reply = self.error_handler()
        self.history.append({"role": "assistant", "content": reply})
        self.save_memory()
        return reply


# Example server integration:
# nova = Nova()
# reply = nova.chat("What is the ISS altitude right now?")


# ---------------------------------------------------------------------------
# Render / HTTP server
# ---------------------------------------------------------------------------

app = Flask(__name__)
nova = Nova()


@app.get("/")
def index():
    return jsonify({
        "service": "Sunday",
        "status": "online",
        "message": "Sunday API is running."
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


if __name__ == "__main__":
    # Local development only. Render starts the app with Gunicorn.
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
