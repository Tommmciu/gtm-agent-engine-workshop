import os
import uuid
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from gtm_agent import gtm_agent as gtm_agent_module


class ToolCallingModel(BaseChatModel):
    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "test-tool-calling-model"

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [
            item.name
            if not isinstance(item, dict)
            else item.get("name") or item.get("function", {}).get("name")
            for item in tools
        ]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class GtmAgentTests(unittest.TestCase):
    def test_agent_uses_only_domain_tools_and_injects_tool_runtime(self):
        model = ToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_current_rep",
                            "args": {},
                            "id": "get-current-rep",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Rep found"),
            ]
        )
        agent = gtm_agent_module.create_agent(
            model=model,
            tools=gtm_agent_module.DOMAIN_TOOLS,
            system_prompt=gtm_agent_module.SYSTEM_PROMPT,
        )

        with patch.object(
            gtm_agent_module.data_service,
            "get_rep",
            return_value={"rep_id": "rep-123", "name": "Casey"},
        ) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Who am I?"}]},
                config={"metadata": {"user_id": "rep-123"}},
            )

        self.assertEqual(
            model.bound_tool_names,
            [
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            ],
        )
        self.assertEqual(get_rep.call_args.args, ("rep-123",))
        self.assertEqual(result["messages"][-1].content, "Rep found")
        self.assertTrue(
            all(
                next(
                    message
                    for message in messages
                    if isinstance(message, SystemMessage)
                ).content
                == gtm_agent_module.SYSTEM_PROMPT
                for messages in model.calls
            )
        )

    def test_run_agent_preserves_invoke_configuration(self):
        generated_run_id = uuid.UUID("0a8c53e8-358d-4567-8dc1-9e8cc00b2bbc")
        with (
            patch.object(gtm_agent_module.uuid, "uuid4", return_value=generated_run_id),
            patch.object(
                gtm_agent_module.gtm_agent,
                "invoke",
                return_value={"messages": [AIMessage(content="Done")]},
            ) as invoke,
        ):
            result = gtm_agent_module.run_agent(
                "Score the prospect",
                user_id="rep-123",
                environment="staging",
                thread_id="thread-456",
            )

        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Score the prospect"}]},
            config={
                "run_name": "GTM Assistant",
                "run_id": generated_run_id,
                "metadata": {
                    "thread_id": "thread-456",
                    "user_id": "rep-123",
                    "environment": "staging",
                    "request_intent": "score_prospect",
                },
            },
        )
        self.assertEqual(result["run_id"], str(generated_run_id))
        self.assertEqual(result["reply"], "Done")


if __name__ == "__main__":
    unittest.main()
