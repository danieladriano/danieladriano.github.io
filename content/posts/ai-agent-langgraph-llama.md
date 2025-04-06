---
date: 2025-03-14
# description: ""
# image: ""
lastmod: 2025-03-14
showTableOfContents: false
# tags: ["",]
title: "Creating an AI Agent with LangGraph and Llama 3.1"
type: "post"
---
In this post, we will create an Agent responsible for handling restaurant reservations. This agent uses a local Llama 3.1 LLM model with tool call capabilities and is built using [LangGraph](https://langchain-ai.github.io/langgraph/).

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for building stateful, multi-actor applications with LLMs. It is used to create agent and multi-agent workflows.

Key Features of LangGraph:

- Graph-Based Execution: LangGraph enables complex workflows where agents interact through loops, branches, and structured flows, moving beyond simple linear sequences.
- Multi-Agent Support: Coordinates multiple AI agents working together, each with specific roles and responsibilities.
- Stateful Execution: Preserves state across interactions, essential for long-running tasks.
- Concurrency: Enables parallel operation of multiple agents through asynchronous execution.
- Integration with LangChain: This integration leveraged LangChain's components (memory, tools, and models) to create more powerful AI systems.

## The start

To start, we need to run the Llama 3.1 LLM model locally. I will use [Ollama](https://ollama.com/) for this project. [Download](https://ollama.com/download) Ollama from the official site to install it. Check out my previous post if you use an AMD card and want to configure Ollama with ROCm.

Due to GPU memory limitations, I will use Llama 3.1 with 8 billion parameters. To pull the model from Ollama:

```bash
ollama pull llama3.1:8b
```

As dependency manager, I'm using [uv](https://docs.astral.sh/uv/), but feel free to use any other you prefer.

Start a new project:

```bash
uv init single-agent
```

Let's install the main dependencies:

```bash
uv add "langchain-ollama>=0.2.3" "langgraph>=0.3.5" "ollama>=0.4.7"
```

With Ollama running Llama 3.1 and the project dependencies installed, we are ready to build the agent.

## The project

The main idea of this project is to create an Agent that can help a restaurant handle reservations. The agent needs to be capable of doing three basic actions:

- List available slot times (now + 5 days)
- Book a table
- Cancel the reservation

For each of these actions, the agent will need a tool. A tool is a function or external resource the agent can invoke to perform specific actions. These can range from simple utilities to complex operations.

Since we will not integrate with any database, let's create a function that mocks a dictionary to store our reservations. First, we get today + 5 days in a list and then get 30-minute slot times from 6 PM.

```python
from datetime import datetime, timedelta

def generate_reservations() -> dict[str, dict[str, str]]:
    dates = [
        (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6)
    ]
    start_time = datetime.strptime("18:00", "%H:%M")
    times = [
        (start_time + timedelta(minutes=30 * i)).strftime("%H:%M") for i in range(8)
    ]

    times_dict = {time: None for time in times}
    book = {date: times_dict for date in dates}

    return book

BOOK = generate_reservations()
```

The final format will be:

```python
{
    "2025-03-12": {
		    "06:00": None,
		    "06:30": None,
		    "07:00": None,
		    ...
    }
}
```

## The Tools

The first tool is `list_time_slots`. As the name suggests, it lists the slots in the `BOOK` dictionary. To properly create a tool, we need to use the `@tool` decorator and add a docstring explaining its objective. The docstring is crucial for the LLM to understand the tool context and when to use it.

```python
from langchain_core.tools import tool

@tool
def list_time_slots(date: str) -> dict:
    """Return time slots for reservations

    Args:
        date (str): date to return the slots. Format YYYY-mm-dd

    Returns:
        dict: available time slots
    """
    return BOOK[date]
```

The following two tools, `book_table` and `cancel_reservation`, are responsible for booking a table on a specific date and time and canceling reservations, respectively.

```python
@tool
def book_table(date: str, time: str, name: str, number_persons: int) -> None:
    """Book a table

    Args:
        date (str): date to book the table. Format YYYY-mm-dd
        time (str): time slot. Format HH:MM
        name (str): name of the person that are booking the table
        number_persons (int): number of persons on the reservation
    """
    BOOK[date][time] = {"name": name, "number_persons": number_persons}

@tool
def cancel_reservation(date: str, time: str) -> None:
    """Cancel a reservation by date and time

    Args:
        date (str): date to cancel the reservation. Format YYYY-mm-dd
        time (str): time slot to cancel. Format HH:MM

    Returns:
        bool: True if the reservation was cancel
    """
    BOOK[date][time] = None
    return True
```

## The Agent

Now that our tools are created, we can start building the agent.
We need to start by creating a `StateGraph`. A `StateGraph` object defines the structure of our chatbot as a "state machine”. The messages attribute has the type `list`, and the `add_messages` function in the annotation defines how this state key should be updated. In this case, it appends messages to the list rather than overwriting them.

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

The `_prompt` property returns a `RunnableCallable` that concatenates a `SystemMessage` with an initial guide prompt for the LLM and the state message that will be received with a `HumanMessage`.

Next, we create our `Agent` class that holds the nodes and conditional routers. The `call_model` function is a node. Nodes represent units of work, and they are typically regular Python functions. As we can see, the `call_model` node function takes the current `State` as input and returns a dictionary containing an updated messages list under the key “messages". This is the basic pattern for all LangGraph node functions. After, we get our prompt and create a chain inside the node with the LLM that binds our tools. The binding tool lets the LLM know the correct JSON format to use if it wants to use our tools. Then, we invoke the created chain, passing the `State` messages as input.

```python
from datetime import datetime
from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import TypedDict

from llm_models import SupportedLLMs, get_llm
from reservations import book_table, cancel_reservation, list_time_slots

class Agent:
    @property
    def _prompt(self) -> RunnableCallable:
        content = f""" You are a helpfull restaurant assistant responsible for reservations.
                    1. Choose your action using the tools that are available to you
                    2. If there is no tool to call with the user request, ask for more context about what the usar want
                    3. Elaborate a response to the user
                                                
                    Name of the restaurant: Tastes of Brazil
                    Current Date: {datetime.now()}
                    """
        system_message = SystemMessage(content=content)
        return RunnableCallable(lambda state: [system_message] + state, name="Prompt")

		def call_model(self, state: State) -> State:
        assistant_runnable = self._prompt | self._llm.bind_tools(
            [list_time_slots, book_table, cancel_reservation]
        )
        response = assistant_runnable.invoke(state["messages"])
        return {"messages": [response]}
```

Next, we must define our conditional router and then build the graph. The `conditional_router` function is a router that analyses the last message received to see if it is a tool call. This tool call is returned by the LLM, which, based on the user input, decides if there is a tool to handle the request. If it is a tool call, return the node's name to be called, which, in that case, is the `tools`. Otherwise, return the `END` saying to the graph to go to the `END` node.

```python
class Agent:
		...
    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
```

To build our agent, we create a `StateGraph` and set the `state_schema` to `State`. Then, we add our `call_model` node and a `ToolNode` with the list of our tools.

The first edge to add is the entry point for our graph, using the `START`node node from LangGraph, which goes to our `call_model` node. Since the LLM defines the output from the `call_model` node (it can be tool calls), we need to define a conditional edge with a source point in our `call_model` node that can go to the tools or `END`. The `conditional_router` function determines the path. Then, we define an edge that creates a return from the tools to the `call_model` and compile our graph.

```python
class Agent:
		...

    def build_agent(self) -> CompiledStateGraph:
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node(node="call_model", action=self.call_model)
        graph_builder.add_node(
            node="tools",
            action=ToolNode([list_time_slots, book_table, cancel_reservation]),
        )

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model", 
            path=self.conditional_router, 
            path_map=["tools", END]
        )
        graph_builder.add_edge(start_key="tools", end_key="agent")

        return graph_builder.compile()
```

## The chat

Now, we can use a chat library, create an API, or, in our case, define logic that uses input and print to simulate a chat between the user and the agent.

```python
def stream_graph_updates(graph: CompiledStateGraph, user_input: str) -> None:
    messages = {"messages": [("user", user_input)]}
    events = graph.invoke(messages, stream_mode="values")
    print(f"Assistant: {events['messages'][-1].content}")

def main() -> None:
    llm = get_llm(llm_model=SupportedLLMs.llama3_1)
    chatbot = Agent(llm=llm)
    graph = chatbot.build_agent()

    print("Assistant: Welcome to Tastes of Brazil! How can I help you today?")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            stream_graph_updates(graph=graph, user_input=user_input)
        except Exception as e:
            print(f"Error {e}")

if __name__ == "__main__":
    main()
```

## The LLM

The final part of our project is the `get_llm` function. Since Ollama is serving the models, we can test the behavior of different models in our agent. This function receives the LLM we want and then creates a `ChatOllama`. The Ollama server will load the model.

```python
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

class SupportedLLMs(Enum):
    llama3_1 = "llama3.1"
    llama3_2 = "llama3.2"
    mistral_7b = "mistral_7b"

def get_llm(llm_model: SupportedLLMs) -> BaseChatModel:
    if llm_model == SupportedLLMs.llama3_1:
        return ChatOllama(model="llama3.1:8b")
    if llm_model == SupportedLLMs.llama3_2:
        return ChatOllama(model="llama3.2")
    if llm_model == SupportedLLMs.mistral_7b:
        return ChatOllama(model="mistral:7b")
    raise Exception("LLM not supported")
```

## The results

We can start the conversation by asking for the available slot times for tomorrow. The LLM in `call_model` node will analyze the user message and identify the most suitable tool to call. In this case, it will be the `list_time_slots`. After the tool execution, the LLM receives the time slots, elaborates a message to the user, and sends it to him.

```
Assistant: Welcome to Tastes of Brazil! How can I help you today?

User: I want to book a table for tomorrow, can you show me the available slots?

Assistant: We have availability for 6:00 PM, 6:30 PM, 7:00 PM, 7:30 PM, 8:00 PM, 
8:30 PM, 9:00 PM, and 9:30 PM tomorrow. Which time slot would you like to book?
```

Next, we informed them that we wanted to book a table for four people tomorrow at 8 PM and put the reservation in Daniel's name. The same steps as before are executed, but the `book_table` tool is called this time. 

```
User: Book the time slot at 8PM of tomorrow.
It will be 4 person and put the reservation in the name of Daniel

Assistant: Your table for 4 people has been booked at 8PM tomorrow,
under the name Daniel. We're looking forward to serving you!
Would you like to make any special requests or note any dietary restrictions?
```

Then, we can list the slot times again and verify that our reservation was in there.

```
User: Can you list the reservations for tomorrow?

Assistant: We have a reservation for tomorrow at 8pm with Daniel and 
three other people. There are no reservations at the other available time slots.
Would you like to make a reservation or ask about a different day?
```

We need to provide the date and the hour to cancel a reservation so the LLM can call the `cancel_reservation` to do that. Then, list again to see that there is no reserve anymore for tomorrow.

```
User: I want to cancel my reservation for tomorrow at 8PM

Assistant: Your reservation has been cancelled.
We're sorry you won't be able to join us tomorrow at 8PM.
If you'd like to make a new reservation, please feel free to call back or
visit our website. Thank you for choosing Tastes of Brazil!

User: Can you list the reservations for tomorrow?

Assistant: We currently have no reservations for tomorrow.
Our tables are available for booking throughout the evening.
Would you like to make a reservation?
```

As we can see, the LLM calls each tool and provides the correct arguments. This is why providing a good docstring in our tools, specifying the tool's main objective and its arguments, is essential.
Sometimes, the LLM can create an answer with more information than necessary. For example, after we book our table, it responds that the reservation is done and asks if we would like to make any special requests or note any dietary restrictions. This part of the answer is a hallucination; we can minimize it by providing a better prompt.

## The conclusion and next steps

With the latest advances in LLM, AI Agents are getting more power and capabilities. In this simple example, we create a node that calls a local Llama 3.1 model with tool call capabilities to handle user requests. The complete code is available in my [GitHub account](https://github.com/danieladriano/single-agent).
In the following steps, we will create a multi-agent chatbot and add memory.
If you have any doubts, you can reach me through LinkedIn. Thanks for reading 😀