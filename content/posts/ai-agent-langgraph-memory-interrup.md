---
date: 2025-04-06
# description: ""
image: "/ai-agent-memory-hil/output_image.png"
lastmod: 2025-04-06
showTableOfContents: false
# tags: ["",]
title: "AI Agent - Adding short-term memory and human-in-the-loop"
type: "post"
---

[In my previous post](https://danieladriano.github.io/posts/ai-agent-langgraph-llama/), we built an agent responsible for handling restaurant reservations using LangGraph and a local Llama 3.1 model. In this post, we changed the idea from a restaurant to a car dealership, but the core implementation remains unchanged, expanding adding short-term memory (checkpointer) and an interrupt to deal with user confirmation.

This agent must handle the list/searching of cars and the scheduling/canceling of test drivers.

## The short-term memory

In short-term memory, our chat history is stored and retrieved based on the chat thread/session. LangGraph deals with this thread-based memory, using a checkpointer that saves changes in the states between nodes in our graph. In this project, we use the `MemorySaver` checkpointer to simplify our project, but LangGraph also has a `PostgresSaver` .

To use the checkpointer, we must pass the `MemorySaver` when compiling the graph.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph.compile(checkpointer=checkpointer)
```

When invoking the graph, you must pass a `RunnableConfiguration` with `thread-id` for this chat. If you use the same `thread-id`, the chat history will be available in each node.

```python
config = {"configurable": {"thread_id": "1"}}
graph.invoke(input=message, config=config, stream_mode="values")
```

This is all that is necessary to use the checkpointer.

In this project, we create an agent class that receives the checkpointer as a parameter in the `build_agent` method. This method adds nodes and edges to our graph and then compiles them.

```python
...
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
...

class Agent:
		...
    def build_agent(
        self, checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        graph_builder = StateGraph(state_schema=State)
        graph_builder.add_node(node="call_model", action=self.call_model)
        graph_builder.add_node(
            node="tools",
            action=ToolNode([list_inventory, list_test_drives, schedule_test_drive]),
        )
        graph_builder.add_node("cancel_test_drive", self.cancel_test_drive_node)

        graph_builder.add_edge(start_key=START, end_key="call_model")
        graph_builder.add_conditional_edges(
            source="call_model",
            path=self.conditional_router,
            path_map=["tools", "cancel_test_drive", END],
        )
        graph_builder.add_edge(start_key="tools", end_key="call_model")
        graph_builder.add_edge(start_key="cancel_test_drive", end_key="call_model")

        return graph_builder.compile(checkpointer=checkpointer)

```

Full Agent Class - https://github.com/danieladriano/car-dealership-agents/blob/main/agent.py

## The Human-in-the-loop - user confirmation using interrup

Human-in-the-loop (HIL) interactions are essential when working with agent systems. Sometimes, we need to confirm information or wait for human input. In LangGraph, we can use the `interrupt()` function to interact with the user in the middle of our graph execution. 

To add `interruptions` in our agents, it's necessary that our graph is being built using a checkpointer. Luckily for us, we already did that 🙂

In this agent, when the user asks to cancel a test drive, he needs to confirm this operation. First, let's create the `tool` and the method that will execute the cancel. In this case, the `tool` is a Pydantict BaseModel `CancelTestDrive` that will be injected into our LLM. The docstring, in this case, is very important since the LLM needs this description to have a context about this tool. Also, we add the `code` field; that way, the LLM will inject the code informed by the user. Our `cancel_test_drive` function receives this `code` and then searches for him inside our `TEST_DRIVE` and then sets the `status` as `TestDriveStatus.CANCEL`

```python
import logging
...
from pydantic import BaseModel, Field

from store.dealership_store import (
    TEST_DRIVE,
    Car,
    TestDrive,
    TestDriveStatus,
    save_test_drivers,
)

logger = logging.getLogger("ai-chat")

class CancelTestDrive(BaseModel):
    """Cancel a test drive"""
    code: int = Field(description="The code of the test drive to cancel")

def cancel_test_drive(code: int) -> bool:
    """Cance a test drive

    Args:
        code (int): The code of the test drive

    Returns:
        bool: Confirm the cancel
    """
    logger.info(f"Cancel test drive of code {code}")
    for test in TEST_DRIVE:
        if test.code == code:
            test.status = TestDriveStatus.CANCEL
            save_test_drivers()
            return True
    return False
 ...
```

Full test_drive module - https://github.com/danieladriano/car-dealership-agents/blob/main/tools/test_drive.py

Now, we need to create a node responsible for confirming with the user if he wants to cancel and then call the `cancel_test_drive` method at `tools/test_drive.py` . In this node, we get the `tool_call["args"]` that the LLM returned, then create a `CancelTestDrive` BaseModel. Then, we call the `interrupt` passing a custom message. This will break the graph execution, returning this message to the user. After the user answers, the graph will resume executing from this node. One heads up is that when the graph resumes his execution, the entire node will be executed again, including the code before the `interrupt`. With the `user_answer` , we verify if he confirmed the cancel operation and then call the `cancel_test_drive` function. Finally, return a `ToolMessage` with the content according to the user confirmation and with the same `tool_call_id` as the `tool_call` from the LLM. This is necessary because every `tool_call` needs a `ToolMessage` as a result.

```python
...

class Agent:
		...
    def cancel_test_drive_node(self, state: State) -> State:
        if isinstance(state["messages"][-1], AIMessage):
            tool_call = state["messages"][-1].tool_calls[0]
        cancel = CancelTestDrive.model_validate(tool_call["args"])
        user_answer = interrupt(
            f"Do you confirm the cancel of test drive code {cancel.code}? [y/n]"
        )
        content = "User gave up canceling, He want to do the test drive."
        if user_answer.content == "y":
            content = (
                "Error when canceling the test drive. Need to call do the dealership."
            )
            if cancel_test_drive(code=cancel.code):
                content = "Test drive canceled."
        return {
            "messages": [
                ToolMessage(content=content, tool_call_id=tool_call["id"], type="tool")
            ]
        }
		...
```

Now, we will create our `conditional_router` called after the `call_model` node. This conditional edge verifies if the LLM returned a `tool_call`. If so, identify by the `name` which tool to call and then return the name of the node responsible for that tool. As we can see, only the `CancelTestDrive` tool call has a specific node, while all other tools are inside the `ToolNode` with the name “tools”.

```python
...

class Agent:
		...
    def conditional_router(self, state: State) -> str:
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.info(
                f"ToolCall - {last_message.tool_calls[0]['name']} - Args {last_message.tool_calls[0]['args']}"
            )
            if last_message.tool_calls[0]["name"] == "CancelTestDrive":
                return "cancel_test_drive"
            return "tools"
        return END
		...
```

In the `call_model` node, we bind the tools available to the LLM.

```python
...

class Agent:
		...
    def call_model(self, state: State) -> State:
        logger.info("Calling model")
        assistant_runnable = self._prompt | self._llm.bind_tools(
            [list_inventory, list_test_drives, schedule_test_drive, CancelTestDrive]
        )
        response = assistant_runnable.invoke(state["messages"])
        return {"messages": [response]}  # type: ignore
		...
```

## The interrupt messages and resuming the graph

When an interrupt occurs, getting the message content back to the user differs slightly. We first need to get the graph state using the config that holds the `thread-id` and then verify if there is any interrupt. To resume our graph execution with the user's answer, we verify that the last state of the graph is an interrupt and then pass a `Command` to the graph with the user message.

```python
def _get_interrupt(
    graph: CompiledStateGraph, config: RunnableConfig
) -> Optional[Interrupt]:
    state = graph.get_state(config=config)
    if state.tasks and state.tasks[0].interrupts:
        return state.tasks[0].interrupts[0]
    return None

def _build_graph_input(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> dict[str, list[tuple[str, str]]] | Command:
    interrupt = _get_interrupt(graph=graph, config=config)
    if interrupt:
        return Command(resume=HumanMessage(content=user_input))
    return {"messages": [("user", user_input)]}

def stream_graph_updates(
    graph: CompiledStateGraph, config: RunnableConfig, user_input: str
) -> str:
    message = _build_graph_input(graph=graph, config=config, user_input=user_input)
    events = graph.invoke(input=message, config=config, stream_mode="values")

    interrupt = _get_interrupt(graph=graph, config=config)
    if interrupt:
        return interrupt.value

    return events["messages"][-1].content
```
Full main module - https://github.com/danieladriano/car-dealership-agents/blob/main/main.py

## The results

Now we can test our agent and see if the memory and the interrupt properly work.

When we ask for a specific car, the model calls the `list_inventory` tool that returns all cars in stock. Then, the LLM filters only for the specific model that we want and returns. As we can see, the answer already suggests to the user to schedule a test drive, informing that it's necessary to pass the car, date and time, name, and driver’s license number.

```
INFO:ai-chat:Assistant: Welcome! How can I help you today?
INFO:ai-chat:================================================================================
INFO:ai-chat:User: Hello, I want to buy a new car. Do you have any golf in stock?
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - list_inventory - Args {'car_model': 'golf'}
INFO:ai-chat:Getting inventory
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: We have several Golf models in stock:

- A black 2025 VW Golf with a price of $35,000.
- A blue 2025 VW Golf priced at $35,500.
- A white 2023 VW Golf that has been driven for 14,867 km and is listed at $23,500.

Would you like to schedule a test drive with one of these cars?
If so, please let me know your preferred date and time as well as
any other necessary details such as your name and driver’s license number.

```

Then, we inform the car, name, driver's license, and date. At this moment, the LLM calls the `schedule_test_drive` tool, but since the car args are wrong (it’s necessary to have all [Car fields](https://github.com/danieladriano/car-dealership-agents/blob/main/store/dealership_store.py#L33)), the LLM keeps trying until it passes the correct arguments. We can see that it calls the `schedule_test_drive` tool four times and also one time the `list_inventory` to have more context about the available cars. 

```
INFO:ai-chat:User: Great. I want to schedule a test drive for the 2025 black one.
                   Name: Daniel Adriano, Drivers license: 123HGJS787,
                   Date: April 07 11 AM
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - schedule_test_drive - Args {'car': 'golf', 'date': '2025-04-07T11:00:00.671376', 'driver_licence': '123HGJS787', 'name': 'Daniel Adriano'}
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - schedule_test_drive - Args {'car': {'model': 'golf', 'year': 2025}, 'date': '2025-04-07T11:00:00.671376', 'driver_licence': '123HGJS787', 'name': 'Daniel Adriano'}
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - list_inventory - Args {'car_model': 'golf'}
INFO:ai-chat:Getting inventory
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - schedule_test_drive - Args {'car': {'color': 'black', 'model': 'golf', 'year': 2025}, 'date': '2025-04-07T11:00:00.671376', 'driver_licence': '123HGJS787', 'name': 'Daniel Adriano'}
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - schedule_test_drive - Args {'car': {'color': 'black', 'kms': 0, 'model': 'golf', 'value': 35000, 'year': 2025}, 'date': '2025-04-07T11:00:00.671376', 'driver_licence': '123HGJS787', 'name': 'Daniel Adriano'}
INFO:ai-chat:Scheduling test drive: code=7 date=datetime.datetime(2025, 4, 7, 11, 0, 0, 671376) car=Car(brand=<Brand.VW: 'Volkswagen'>, model=<Models.GOLF: 'golf'>, color=<Color.BLACK: 'black'>, kms=0, year=2025, value=35000.0) name='Daniel Adriano' driver_licence='123HGJS787' status=<TestDriveStatus.SCHEDULED: 'scheduled'>
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: Your test drive for the 2025 black Volkswagen Golf is successfully scheduled for April 07 at 11 AM. Your test drive code is **7**.

If you need to make any changes or have further questions, feel free to ask! Enjoy your test drive and let us know if there’s anything we can assist with today.
INFO:ai-chat:================================================================================
INFO:ai-chat:User: Thanks
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: You're welcome! If you need any more assistance or have additional questions, don't hesitate to reach out. Have a great day and enjoy your test drive!
INFO:ai-chat:================================================================================
```

To confirm that our test drives are scheduled, we can go to `./store/test_driver.json` file and see a new register with code 7.

```json
[
  {
    "code": 5,
    "date": "2025-04-08T11:59:56.375796",
    "car": {
      "brand": "Volkswagen",
      "model": "golf",
      "color": "black",
      "kms": 0,
      "year": 2025,
      "value": 35000
    },
    "name": "John Doe",
    "driver_licence": "HGJSK123KO",
    "status": "scheduled"
  },
  {
    "code": 7,
    "date": "2025-04-07T11:00:00.671376",
    "car": {
      "brand": "Volkswagen",
      "model": "golf",
      "color": "black",
      "kms": 0,
      "year": 2025,
      "value": 35000
    },
    "name": "Daniel Adriano",
    "driver_licence": "123HGJS787",
    "status": "scheduled"
  }
]
```

Now that our test drives are scheduled, let’s try to cancel. After being informed that we want to cancel a test drive, the LLM identifies that the `code` argument is necessary to call the `CancelTestDrive` tool. Then, replies to the user asking for the code. We just send “Code 7”, then the LLM will correctly call the `CancelTestDrive` tool, which will execute the `cancel_test_drive` node. In this node, we have the `interrupt` that asks the user to confirm the cancel operation.

```
INFO:ai-chat:Assistant: Welcome! How can I help you today?
INFO:ai-chat:================================================================================
INFO:ai-chat:User: Hello, I want to cancel my test drive.
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: Of course, I can help you with that. Could you please provide me with the code of your scheduled test drive?
INFO:ai-chat:================================================================================
INFO:ai-chat:User: Code 7
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - CancelTestDrive - Args {'code': 7}
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: Do you confirm the cancel of test drive code 7? [y/n]
INFO:ai-chat:================================================================================
INFO:ai-chat:User: y
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Cancel test drive of code 7
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: Your test drive has been successfully canceled. If you have any other requests or need further assistance, feel free to let me know!
INFO:ai-chat:================================================================================
```

To test short-term memory, we can ask for a specific car model and then ask for another one. Since the `list_inventory` tool is called the first time and returns all available car models, when we ask for a different model, the LLM will not call the `list_inventory` tool again since it has the list of cars in memory.

```
INFO:ai-chat:Assistant: Welcome! How can I help you today?
INFO:ai-chat:================================================================================
INFO:ai-chat:User: Hello, I want to buy a new car. Do you have any golf in stock?
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:ToolCall - list_inventory - Args {}
INFO:ai-chat:Getting inventory
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: We have several Golf models in stock:

- 2025 VW Golf, Black, Kms: 0, Price: $35,000
- 2025 VW Golf, Blue, Kms: 0, Price: $35,500
- 2023 VW Golf, White, Kms: 14867, Price: $23,500

Would you like to schedule a test drive for any of these models?
INFO:ai-chat:================================================================================
INFO:ai-chat:User: Great. Do you have any used polo?
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Calling model
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:ai-chat:--------------------------------------------------------------------------------
INFO:ai-chat:Assistant: Here are the used Polo models in stock:

- 2022 VW Polo, Red, Kms: 9239, Price: $20,000

Would you like to schedule a test drive for this car or do you want information on other available Polos?
INFO:ai-chat:================================================================================
```

## The conclusion

In this post, we create an Agent who helps a car dealership handle listing/searching cars and test driving. We added a short-term memory in the graph, allowing memory between chats in the same thread and also interrupting the graph execution to talk with the user. You can get the entire project code at my [GitHub](https://github.com/danieladriano/car-dealership-agents/tree/main).

If you have any doubts, you can reach me through LinkedIn. Thanks for reading 😀