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


def search_movies(genre: str) -> str:
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
    }
    return movies.get(genre.lower(), "No movies found for that genre.")


def change__to_f(temp_c: float) -> float:
    return temp_c * 1.8 + 32


def get_weather(city: str) -> str:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=15,
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
        timeout=15,
    )
    weather.raise_for_status()
    current = weather.json()["current"]

    return json.dumps({
        "resolved_city": location["name"],
        "temperature_celsius": current["temperature_2m"],
        "weather_code": current["weather_code"],
    })


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


class CrewState(TypedDict):
    input: str
    messages: List[Any]
    developer_output: Optional[str]
    tester_output: Optional[str]
    manager_output: Optional[str]
    final_output: Optional[str]


def task_input_node(state: CrewState):
    return {"messages": [HumanMessage(content=state["input"])]}


def developer_node(state: CrewState):
    user_input = state["input"].strip()
    lower_input = user_input.lower()

    if any(w in lower_input for w in ["weather", "temperature", "temp", "forecast"]):
        match = re.search(
            r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z .'-]*?)(?:\?|$)",
            user_input,
            re.IGNORECASE,
        )
        city = match.group(1).strip() if match else user_input
        city = re.sub(
            r"\s+(today|now|currently|right now)$",
            "",
            city,
            flags=re.IGNORECASE,
        ).strip()

        try:
            data = json.loads(get_weather(city))
            descriptions = {
                0: "clear sky", 1: "mainly clear", 2: "partly cloudy",
                3: "overcast", 45: "foggy", 48: "foggy",
                51: "light drizzle", 53: "moderate drizzle",
                55: "heavy drizzle", 61: "light rain",
                63: "moderate rain", 65: "heavy rain",
                71: "light snow", 73: "moderate snow", 75: "heavy snow",
                80: "light rain showers", 81: "moderate rain showers",
                82: "heavy rain showers", 95: "thunderstorm",
                96: "thunderstorm with hail", 99: "thunderstorm with hail",
            }
            city_name = data.get("resolved_city", city)
            temp = data.get("temperature_celsius")
            desc = descriptions.get(data.get("weather_code"), "current conditions")
            answer = f"Current weather in {city_name}: {temp}°C, {desc}."
        except Exception as exc:
            answer = f"Unable to retrieve weather information: {exc}"

        return {"developer_output": answer}

    if any(w in lower_input for w in ["movie", "movies", "film", "films", "cinema"]):
        genre = "comedy"
        for possible_genre in ["sci-fi", "comedy", "action"]:
            if possible_genre in lower_input:
                genre = possible_genre
                break
        return {
            "developer_output":
            f"Here are some Indian {genre} movies: {search_movies(genre)}"
        }

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

    if isinstance(response.content, list):
        answer = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in response.content
        ).strip()
    else:
        answer = str(response.content).strip()

    if not answer or answer == ".":
        answer = "I am not authorized to answer questions outside of Indian weather and cinema."

    return {"developer_output": answer}


def tester_node(state: CrewState):
    if state.get("developer_output") and str(state["developer_output"]).strip():
        result = "PASS: Developer response generated successfully."
    else:
        result = "FAIL: Developer response was empty."
    return {"tester_output": result}


def manager_node(state: CrewState):
    answer = state.get("developer_output", "")
    return {"manager_output": answer, "final_output": answer}


workflow = StateGraph(CrewState)
workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "task_input")
workflow.add_edge("task_input", "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

graph = workflow.compile()


class AgentInput(BaseModel):
    input: str = Field(description="Message sent to the agent.")


def format_for_graph(x):
    user_input = x.get("input", "") if isinstance(x, dict) else x.input
    return {
        "input": str(user_input),
        "messages": [],
        "developer_output": None,
        "tester_output": None,
        "manager_output": None,
        "final_output": None,
    }


def format_graph_output(state):
    if not isinstance(state, dict):
        return str(state)
    final_output = state.get("final_output")
    if final_output and str(final_output).strip():
        return str(final_output)
    manager_output = state.get("manager_output")
    if manager_output and str(manager_output).strip():
        return str(manager_output)
    return "No output was generated."


formatted_agent_chain = (
    RunnableLambda(format_for_graph)
    | graph
    | RunnableLambda(format_graph_output)
).with_types(input_type=AgentInput, output_type=str)


app = FastAPI(
    title="Movie and Weather Agent",
    version="1.0",
    description="LangGraph workflow: Task Input -> Developer -> Tester -> Manager",
)


@app.get("/")
def root():
    return {"message": "Server is running. Open /agent/playground/ to use the agent."}


add_routes(app, formatted_agent_chain, path="/agent")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
