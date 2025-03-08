import openai
from .utils import compare_toolname, convert_mcp_tools_to_openai_format
import os
from mcp.client.sse import sse_client
from mcp import ClientSession
from typing import Dict, Any
import logging
import json
logger = logging.getLogger(__name__)


async def prompt(user_message: str) -> str:
    llm_client = openai.AsyncClient(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY")
    )

    model_id = os.getenv("LLM_MODEL_ID")

    messages = [
        {
            'role': 'user',
            'content': user_message
        }
    ]

    async with sse_client(
        f"http://{MCP_SSE_HOST}:{MCP_SSE_PORT}/sse"
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            
            if hasattr(mcp_tools, 'tools'):
                mcp_tools = mcp_tools.tools
            
            openai_compatible_tools = convert_mcp_tools_to_openai_format(
                mcp_tools
            )
            
            async def execute_openai_compatible_toolcall(
                toolname: str, arguments: Dict[str, Any]
            ) -> str:
                nonlocal mcp_tools, session

                actual = [
                    tool.name
                    for tool in mcp_tools
                    if compare_toolname(toolname, tool.name)
                ]
                
                if len(actual) > 1:
                    logger.warning(
                        "More than one tool has the same santizied"
                        " name to the required tool"
                    )

                elif len(actual) == 0:
                    return f"Tool {toolname} not found"

                toolname = actual[0]
                res = await session.call_tool(toolname, arguments)

                if res.isError:
                    return (
                        f"Something went wrong while "
                        f"executing tool {toolname} with {arguments}")

                return res.content 
            
            completion = await llm_client.chat.completions.create(
                messages=messages,
                model=model_id,
                tools=openai_compatible_tools
            )
            
            messages.append(completion.choices[0].message.model_dump())
            logger.info(
                f"Assistant: {completion.choices[0].message.model_dump()}"
            )

            requested_toolcalls = (
                completion.choices[0].message.tool_calls or []
            )

            for call in requested_toolcalls:
                _id, _name = call.id, call.function.name
                _args = json.loads(call.function.arguments)

                logger.info(f"* Calling {_name}: {_args}")
                result = await execute_openai_compatible_toolcall(_name, _args)
                logger.info(f"* Got: {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _id,
                        "content": result
                    }
                )

            if len(requested_toolcalls) > 0:
                post_completion = await llm_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=openai_compatible_tools,
                )
                
                messages.append(
                    {
                        "role": "assistant",
                        "content": post_completion.choices[0].message.content
                    }
                )

                logger.info(
                    f"Assistant: {post_completion.choices[0].message.content}"
                )

    return messages[-1]['content']