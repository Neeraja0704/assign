import os
import json
import re
import uvicorn
import requests

from typing import TypedDict, List, Optional, Any

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


# ============================================================
# 1. TOOLS
# ============================================================

def search_movies(genre: str) -> str:
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre."
    )


def change__to_f(temp_c: float) -> float:
    return temp_c * 1.8 + 32


def get_weather(city: str) -> str:

    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1
        },
        timeout=15
    )

    geo.raise_for_status()

    geo_data = geo.json()

    if "results" not in geo_data or not geo_data["results"]:
        return f"Could not find weather data for city: {city}"

    location = geo_data["results"][0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius",
        },
        timeout=15
    )

    weather.raise_for_status()

    current = weather.json()["current"]

    return json.dumps({
        "resolved_city": location["name"],
        "temperature_celsius": current["temperature_2m"],
        "weather_code": current["weather_code"],
    })


# ============================================================
# 2. GEMINI LLM
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict):

    input: str

    messages: List[Any]

    developer_output: Optional[str]

    tester_output: Optional[str]

    manager_output: Optional[str]


# ============================================================
# 4. TASK INPUT NODE
# ============================================================

def task_input_node(state: CrewState):

    return {
        "messages": [
            HumanMessage(
                content=state["input"]
            )
        ]
    }


# ============================================================
# 5. DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    user_input = state["input"].strip()

    lower_input = user_input.lower()


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    if any(
        word in lower_input
        for word in [
            "weather",
            "temperature",
            "temp",
            "forecast"
        ]
    ):

        match = re.search(
            r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z .'-]*?)(?:\?|$)",
            user_input,
            re.IGNORECASE
        )

        city = (
            match.group(1).strip()
            if match
            else user_input
        )

        city = re.sub(
            r"\s+(today|now|currently|right now)$",
            "",
            city,
            flags=re.IGNORECASE
        ).strip()


        try:

            data = json.loads(
                get_weather(city)
            )


            descriptions = {

                0: "clear sky",

                1: "mainly clear",

                2: "partly cloudy",

                3: "overcast",

                45: "foggy",

                48: "foggy",

                51: "light drizzle",

                53: "moderate drizzle",

                55: "heavy drizzle",

                61: "light rain",

                63: "moderate rain",

                65: "heavy rain",

                71: "light snow",

                73: "moderate snow",

                75: "heavy snow",

                80: "light rain showers",

                81: "moderate rain showers",

                82: "heavy rain showers",

                95: "thunderstorm",

                96: "thunderstorm with hail",

                99: "thunderstorm with hail",

            }


            city_name = data.get(
                "resolved_city",
                city
            )

            temperature = data.get(
                "temperature_celsius"
            )

            description = descriptions.get(
                data.get("weather_code"),
                "current conditions"
            )


            answer = (
                f"Current weather in "
                f"{city_name}: "
                f"{temperature}°C, "
                f"{description}."
            )


        except Exception as exc:

            answer = (
                "Unable to retrieve weather "
                f"information: {exc}"
            )


        return {
            "developer_output": answer
        }


    # --------------------------------------------------------
    # MOVIES
    # --------------------------------------------------------

    if any(
        word in lower_input
        for word in [
            "movie",
            "movies",
            "film",
            "films",
            "cinema"
        ]
    ):

        genre = "comedy"


        for possible_genre in [
            "sci-fi",
            "comedy",
            "action"
        ]:

            if possible_genre in lower_input:

                genre = possible_genre

                break


        answer = (
            f"Here are some Indian "
            f"{genre} movies: "
            f"{search_movies(genre)}"
        )


        return {
            "developer_output": answer
        }


    # --------------------------------------------------------
    # LLM FALLBACK
    # --------------------------------------------------------

    prompt = f"""
You are the Developer state of a multi-state application.

The application is authorized to answer questions about:

1. Indian weather
2. Indian cinema/movies

User request:

{user_input}

If the request is outside these areas, respond exactly:

I am not authorized to answer questions outside of Indian weather and cinema.
"""


    response = llm_flash.invoke(prompt)


    if isinstance(
        response.content,
        list
    ):

        answer = "\n".join(

            str(
                item.get("text", "")
            )
            if isinstance(item, dict)
            else str(item)

            for item in response.content

        ).strip()


    else:

        answer = str(
            response.content
        ).strip()


    if not answer or answer == ".":

        answer = (
            "I am not authorized to "
            "answer questions outside "
            "of Indian weather and cinema."
        )


    return {
        "developer_output": answer
    }


# ============================================================
# 6. TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    developer_output = state.get(
        "developer_output"
    )


    if (
        developer_output
        and str(developer_output).strip()
    ):

        result = (
            "PASS: Developer response "
            "generated successfully."
        )

    else:

        result = (
            "FAIL: Developer response "
            "was empty."
        )


    return {
        "tester_output": result
    }


# ============================================================
# 7. MANAGER NODE
# ============================================================

def manager_node(state: CrewState):

    answer = state.get(
        "developer_output",
        ""
    )


    return {
        "manager_output": answer
    }


# ============================================================
# 8. LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(
    CrewState
)


# Add nodes

workflow.add_node(
    "task_input",
    task_input_node
)

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "manager",
    manager_node
)


# ------------------------------------------------------------
# CONTROL FLOW
# ------------------------------------------------------------

workflow.add_edge(
    START,
    "task_input"
)

workflow.add_edge(
    "task_input",
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    "manager"
)

workflow.add_edge(
    "manager",
    END
)


# Compile graph

graph = workflow.compile()


# ============================================================
# 9. LANGSERVE INPUT
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Message sent to the agent."
    )


# ============================================================
# 10. FORMAT INPUT
# ============================================================

def format_for_graph(x):

    if isinstance(x, dict):

        user_input = x.get(
            "input",
            ""
        )

    else:

        user_input = x.input


    return {

        "input": str(
            user_input
        ),

        "messages": [],

        "developer_output": None,

        "tester_output": None,

        "manager_output": None,

    }


# ============================================================
# 11. FORMAT FINAL OUTPUT
# ============================================================

def format_graph_output(state):

    if not isinstance(state, dict):

        return str(state)


    # IMPORTANT:
    # ONLY manager_output is returned.

    return state.get(
        "manager_output",
        ""
    )


# ============================================================
# 12. LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_graph
    )

    | graph

    | RunnableLambda(
        format_graph_output
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 13. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Movie and Weather Agent",

    version="1.0",

    description=(
        "LangGraph workflow: "
        "Task Input -> Developer -> "
        "Tester -> Manager"
    ),

)


# ============================================================
# 14. HOME ROUTE
# ============================================================

@app.get("/")
def root():

    return {

        "message":
        "Server is running. "
        "Open /agent/playground/ "
        "to use the agent."

    }


# ============================================================
# 15. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent"

)


# ============================================================
# 16. RENDER STARTUP
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )


    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port

    )
