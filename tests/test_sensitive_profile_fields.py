import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LANGSMITH_TRACING", "false")

data_service = importlib.import_module("gtm_agent.data_service")
agent_module = importlib.import_module("gtm_agent.gtm_agent")


class SensitiveProfileFieldTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def tearDown(self):
        data_service._PROFILES.clear()

    def test_get_prospect_returns_only_contact_and_guard_fields(self):
        record = {
            "name": "Avery Example",
            "email": "avery@example.com",
            "disqualified": True,
            "annual_revenue": 1000000,
            "billing_qualification": {"tax_id": "123"},
            "engagement_history": ["meeting"],
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = agent_module.get_prospect.func("LEAD-1")

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-1",
                "name": "Avery Example",
                "email": "avery@example.com",
                "disqualified": True,
            },
        )
        self.assertNotIn("billing_qualification", result["prospect"])

    def test_get_prospect_defaults_missing_disqualification_to_false(self):
        with patch.object(
            data_service,
            "get_prospect_record",
            return_value={"name": "Avery Example", "email": "avery@example.com"},
        ):
            result = agent_module.get_prospect.func("LEAD-1")

        self.assertFalse(result["prospect"]["disqualified"])

    def test_new_profile_excludes_billing_qualification(self):
        prospect = {
            "name": "Avery Example",
            "annual_revenue": 1000000,
            "billing_qualification": {"tax_id": "123"},
        }
        with (
            patch.object(
                data_service,
                "get_profile_from_db",
                return_value={"prospect_profile": None},
            ),
            patch.object(data_service, "get_prospect_record", return_value=prospect),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value={}),
            patch.object(data_service, "fetch_tech_stack", return_value=[]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = agent_module.build_prospect_profile.func("LEAD-1")

        profile = result["prospect_profile"]
        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("billing_qualification", save_profile.call_args.args[1])

    def test_cached_profile_is_sanitized_and_updated(self):
        data_service._PROFILES["LEAD-1"] = {
            "prospect_id": "LEAD-1",
            "billing_qualification": {"tax_id": "123"},
            "tech_stack": ["Python"],
        }

        result = agent_module.build_prospect_profile.func("LEAD-1")

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn(
            "billing_qualification",
            data_service.get_profile_from_db("LEAD-1")["prospect_profile"],
        )
        self.assertNotIn("billing_qualification", data_service._PROFILES["LEAD-1"])

    def test_scoring_prompt_contains_only_scoring_profile_fields(self):
        profile = {
            "prospect_id": "LEAD-1",
            "annual_revenue": 1000000,
            "tech_stack": ["Old"],
            "account_details": {"industry": "Software"},
            "engagement_history": ["meeting"],
            "billing_qualification": {"tax_id": "123"},
            "disqualified": True,
            "name": "Avery Example",
        }
        offering = {
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 500000,
            "description": "Test offering",
        }
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Python"]),
            patch.object(agent_module, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = SimpleNamespace(
                model_dump=lambda: {"score": 90}
            )
            agent_module.score_prospect.func(profile, offering)

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        serialized_profile = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            set(serialized_profile),
            {"annual_revenue", "tech_stack", "account_details", "engagement_history"},
        )
        self.assertEqual(serialized_profile["tech_stack"], ["Python"])
        self.assertNotIn("billing_qualification", user_message)


if __name__ == "__main__":
    unittest.main()
