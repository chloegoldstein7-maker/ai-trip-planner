from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
import boto3


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE", "0") == "1":
        print("Using TEST_MODE - returning fake LLM")
        return _Fake()
    
    # Check for AWS Bedrock configuration
    model_arn = os.getenv("AWS_MODEL_ARN")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    
    print(f"AWS_MODEL_ARN: {model_arn}")
    print(f"AWS_ACCESS_KEY_ID: {access_key[:10]}..." if access_key else "AWS_ACCESS_KEY_ID: None")
    print(f"TEST_MODE: {os.getenv('TEST_MODE', '0')}")
    
    if model_arn and access_key:
        # Create boto3 session with credentials from environment
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-2")
        )
        
        try:
            # For inference profiles, we need to be very explicit about the model_id
            print(f"Initializing ChatBedrock with inference profile: {model_arn}")
            
            bedrock_client = session.client("bedrock-runtime")
            
            # Extract the model name from the ARN to determine provider
            # ARN format: arn:aws:bedrock:region:account:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0
            if "anthropic" in model_arn.lower() or "claude" in model_arn.lower():
                provider = "anthropic"
            else:
                provider = None  # Let it auto-detect for other providers
            
            # Create ChatBedrock with explicit model_id as the inference profile ARN
            llm = ChatBedrock(
                model_id=model_arn,  # This should be the full inference profile ARN
                client=bedrock_client,
                model_kwargs={
                    "temperature": 0.7,
                    "max_tokens": 1500,
                    "anthropic_version": "bedrock-2023-05-31"
                },
                # For inference profiles with Anthropic models, we need to specify the provider
                provider=provider,
                # Ensure streaming is disabled for compatibility
                streaming=False,
                # Additional settings for better compatibility
                verbose=True
            )
            
            print(f"Successfully initialized ChatBedrock with model_id: {model_arn}, provider: {provider}")
            return llm
            
        except Exception as e:
            print(f"Failed to initialize ChatBedrock with {model_arn}: {e}")
            raise ValueError(f"Failed to initialize Bedrock model: {e}")
    else:
        # Require a key unless running tests
        raise ValueError("Please set AWS_MODEL_ARN and AWS credentials in your .env file")


# Initialize LLM lazily to ensure it uses current configuration
def get_llm():
    return _init_llm()


# Minimal tools (deterministic for tutorials)
@tool
def essential_info(destination: str) -> str:
    """Return essential destination info like weather, sights, and etiquette."""
    # Enhanced mock data with actual structure
    return f"""Essential Information for {destination}:
    - Climate: Tropical/temperate with seasonal variations
    - Best time to visit: Spring and fall months
    - Top attractions: Historical sites, natural landmarks, cultural centers
    - Local customs: Respectful dress at religious sites, tipping culture varies
    - Language: Local language with English widely spoken in tourist areas
    - Currency: Local currency, credit cards accepted in most establishments
    - Safety: Generally safe for tourists, standard precautions advised"""


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Return high-level budget categories for a given destination and duration."""
    return f"""Budget breakdown for {destination} ({duration}):
    - Accommodation: $50-200/night depending on style
    - Meals: $30-80/day (street food to restaurants)
    - Local transport: $10-30/day (public transit to taxis)
    - Activities/attractions: $20-60/day
    - Shopping/extras: $20-50/day
    Total estimated daily budget: $130-420 depending on travel style"""


@tool
def local_flavor(destination: str, interests: Optional[str] = None) -> str:
    """Suggest authentic local experiences matching optional interests."""
    interest_str = f" focusing on {interests}" if interests else ""
    return f"""Authentic local experiences in {destination}{interest_str}:
    - Morning markets with local vendors and fresh produce
    - Traditional cooking classes with local families
    - Neighborhood walking tours off the beaten path
    - Local artisan workshops and craft demonstrations
    - Community cultural performances and festivals
    - Hidden cafes and restaurants favored by locals
    - Sacred sites and temples with cultural significance"""


@tool
def day_plan(destination: str, day: int) -> str:
    """Return a simple day plan outline for a specific day number."""
    return f"Day {day} in {destination}: breakfast, highlight visit, lunch, afternoon walk, dinner."


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def weather_brief(destination: str) -> str:
    """Return a brief weather summary for planning purposes."""
    return f"""Weather overview for {destination}:
    - Current season: Varies by hemisphere and elevation
    - Temperature range: 20-30°C (68-86°F) typical
    - Rainfall: Seasonal patterns, pack rain gear if visiting in wet season
    - Humidity: Moderate to high in tropical areas
    - Pack: Layers, sun protection, comfortable walking shoes"""


@tool
def visa_brief(destination: str) -> str:
    """Return a brief visa guidance placeholder for tutorial purposes."""
    return f"Visa guidance for {destination}: check your nationality's embassy site."


@tool
def attraction_prices(destination: str, attractions: Optional[List[str]] = None) -> str:
    """Return rough placeholder prices for attractions."""
    items = attractions or ["Museum", "Historic Site", "Viewpoint"]
    priced = "\n    - ".join(f"{a}: $10-40 per person" for a in items)
    return f"""Attraction pricing in {destination}:
    - {priced}
    - Multi-day passes: Often 20-30% savings
    - Student/senior discounts: Usually 25-50% off
    - Free days: Many museums offer free entry certain days/hours
    - Booking online: Can save 10-15% vs gate prices"""


@tool
def local_customs(destination: str) -> str:
    """Return simple etiquette reminders for the destination."""
    return f"Customs in {destination}: be polite, modest dress in sacred places, learn greetings."


@tool
def hidden_gems(destination: str) -> str:
    """Return a few off-the-beaten-path ideas."""
    return f"""Hidden gems in {destination}:
    - Secret sunrise viewpoint known mainly to locals
    - Family-run restaurant with no sign (ask locals for directions)
    - Abandoned temple/building with incredible architecture
    - Local swimming hole or beach away from tourist crowds  
    - Artisan quarter where craftsmen still work traditionally
    - Night market that only operates certain days
    - Community garden or park perfect for picnics"""


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return an approximate travel time placeholder."""
    return f"Travel from {from_location} to {to_location} by {mode}: ~20-60 minutes."


@tool
def packing_list(destination: str, duration: str, activities: Optional[List[str]] = None) -> str:
    """Return a generic packing list summary."""
    acts = ", ".join(activities or ["walking", "sightseeing"]) 
    return f"Packing for {destination} ({duration}): comfortable shoes, layers, adapter; for {acts}."


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


def research_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    
    # Manually call tools and format the information
    essential = essential_info.invoke({"destination": destination})
    weather = weather_brief.invoke({"destination": destination})
    visa = visa_brief.invoke({"destination": destination})
    
    # Create a comprehensive prompt with all the information
    prompt_t = (
        "You are a research assistant. Based on the following information about {destination}, "
        "provide a comprehensive summary for the traveler:\n\n"
        "Essential Information:\n{essential}\n\n"
        "Weather Information:\n{weather}\n\n"
        "Visa Information:\n{visa}\n\n"
        "Please synthesize this information into a helpful summary."
    )
    vars_ = {
        "destination": destination,
        "essential": essential,
        "weather": weather,
        "visa": visa
    }
    
    # Use HumanMessage instead of SystemMessage for better compatibility
    messages = [HumanMessage(content=prompt_t.format(**vars_))]
    
    calls = [
        {"agent": "research", "tool": "essential_info", "args": {"destination": destination}},
        {"agent": "research", "tool": "weather_brief", "args": {"destination": destination}},
        {"agent": "research", "tool": "visa_brief", "args": {"destination": destination}}
    ]
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = get_llm().invoke(messages)
    
    out = res.content

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination, duration = req["destination"], req["duration"]
    budget = req.get("budget", "moderate")
    
    # Manually call tools and format the information
    budget_info = budget_basics.invoke({"destination": destination, "duration": duration})
    attractions = attraction_prices.invoke({"destination": destination})
    
    # Create a comprehensive prompt with all the information
    prompt_t = (
        "You are a budget analyst. Based on the following information, "
        "create a detailed budget breakdown for {duration} in {destination} with a {budget} budget:\n\n"
        "Budget Overview:\n{budget_info}\n\n"
        "Attraction Prices:\n{attractions}\n\n"
        "Please provide a comprehensive budget plan with daily breakdowns and money-saving tips."
    )
    vars_ = {
        "destination": destination,
        "duration": duration,
        "budget": budget,
        "budget_info": budget_info,
        "attractions": attractions
    }
    
    # Use HumanMessage instead of SystemMessage for better compatibility
    messages = [HumanMessage(content=prompt_t.format(**vars_))]
    
    calls = [
        {"agent": "budget", "tool": "budget_basics", "args": {"destination": destination, "duration": duration}},
        {"agent": "budget", "tool": "attraction_prices", "args": {"destination": destination}}
    ]
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = get_llm().invoke(messages)
    
    out = res.content

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    interests = req.get("interests", "local culture")
    travel_style = req.get("travel_style", "standard")
    
    # Manually call tools and format the information
    local_exp = local_flavor.invoke({"destination": destination, "interests": interests})
    customs = local_customs.invoke({"destination": destination})
    gems = hidden_gems.invoke({"destination": destination})
    
    # Create a comprehensive prompt with all the information
    prompt_t = (
        "You are a local guide. Based on the following information about {destination}, "
        "create a curated list of authentic experiences for someone interested in {interests} "
        "with a {travel_style} travel approach:\n\n"
        "Local Experiences:\n{local_exp}\n\n"
        "Local Customs:\n{customs}\n\n"
        "Hidden Gems:\n{gems}\n\n"
        "Please provide personalized recommendations that match their interests and travel style."
    )
    vars_ = {
        "destination": destination,
        "interests": interests,
        "travel_style": travel_style,
        "local_exp": local_exp,
        "customs": customs,
        "gems": gems
    }
    
    # Use HumanMessage instead of SystemMessage for better compatibility
    messages = [HumanMessage(content=prompt_t.format(**vars_))]
    
    calls = [
        {"agent": "local", "tool": "local_flavor", "args": {"destination": destination, "interests": interests}},
        {"agent": "local", "tool": "local_customs", "args": {"destination": destination}},
        {"agent": "local", "tool": "hidden_gems", "args": {"destination": destination}}
    ]
    
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = get_llm().invoke(messages)
    
    out = res.content

    return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}


def itinerary_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    travel_style = req.get("travel_style", "standard")
    prompt_t = (
        "Create a {duration} itinerary for {destination} ({travel_style}).\n\n"
        "Inputs:\nResearch: {research}\nBudget: {budget}\nLocal: {local}\n"
    )
    vars_ = {
        "duration": duration,
        "destination": destination,
        "travel_style": travel_style,
        "research": (state.get("research") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "local": (state.get("local") or "")[:400],
    }
    with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
        res = get_llm().invoke([HumanMessage(content=prompt_t.format(**vars_))])
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research_node", research_agent)
    g.add_node("budget_node", budget_agent)
    g.add_node("local_node", local_agent)
    g.add_node("itinerary_node", itinerary_agent)

    # Run research, budget, and local agents in parallel
    g.add_edge(START, "research_node")
    g.add_edge(START, "budget_node")
    g.add_edge(START, "local_node")
    
    # All three agents feed into the itinerary agent
    g.add_edge("research_node", "itinerary_node")
    g.add_edge("budget_node", "itinerary_node")
    g.add_edge("local_node", "itinerary_node")
    
    g.add_edge("itinerary_node", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-trip-planner"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="ai-trip-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):

    graph = build_graph()
    # Only include necessary fields in initial state
    # Agent outputs (research, budget, local, final) will be added during execution
    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "tool_calls": [],
    }
    # No config needed without checkpointer
    out = graph.invoke(state)
    return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
