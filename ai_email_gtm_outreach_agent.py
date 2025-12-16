import json
import os
import sys
from typing import Any, Dict, List, Optional

import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools

import prompts


def require_env(var_name: str) -> None:
    if not os.getenv(var_name):
        print(f"Error: {var_name} not set. export {var_name}=...")
        sys.exit(1)


def create_company_finder_agent() -> Agent:
    exa_tools = ExaTools(category="company")
    db = SqliteDb(db_file="tmp/gtm_outreach.db")
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[exa_tools],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_company_finder",
        debug_mode=True,
        instructions=prompts.COMPANY_FINDER_INSTRUCTIONS,
    )


def create_contact_finder_agent() -> Agent:
    exa_tools = ExaTools()
    db = SqliteDb(db_file="tmp/gtm_outreach.db")
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[exa_tools],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_contact_finder",
        debug_mode=True,
        instructions=prompts.CONTACT_FINDER_INSTRUCTIONS,
    )


def create_email_writer_agent(style_key: str = "Professional") -> Agent:
    db = SqliteDb(db_file="tmp/gtm_outreach.db")
    instructions = prompts.get_email_writer_instructions(style_key)
    return Agent(
        model=OpenAIChat(id="gpt-5"),
        tools=[],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_email_writer",
        debug_mode=False,
        instructions=instructions,
    )


def create_research_agent() -> Agent:
    """Agent to gather interesting insights from company websites and Reddit."""
    exa_tools = ExaTools()
    db = SqliteDb(db_file="tmp/gtm_outreach.db")
    return Agent(
        model=OpenAIChat(id="gpt-5"),
        tools=[exa_tools],
        db=db,
        enable_user_memories=True,
        add_history_to_context=True,
        num_history_runs=6,
        session_id="gtm_outreach_researcher",
        debug_mode=True,
        instructions=prompts.RESEARCH_AGENT_INSTRUCTIONS,
    )


def extract_json_or_raise(text: str) -> Dict[str, Any]:
    """Extract JSON from a model response. Assumes the response is pure JSON."""
    try:
        return json.loads(text)
    except Exception as e:
        # Try to locate a JSON block if extra text snuck in
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)
        raise ValueError(f"Failed to parse JSON: {e}\nResponse was:\n{text}")


def run_company_finder(agent: Agent, target_desc: str, offering_desc: str, max_companies: int) -> List[Dict[str, str]]:
    prompt = prompts.COMPANY_FINDER_PROMPT_TEMPLATE.format(
        max_companies=max_companies,
        target_desc=target_desc,
        offering_desc=offering_desc
    )
    resp: RunOutput = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    companies = data.get("companies", [])
    return companies[: max(1, min(max_companies, 10))]


def run_contact_finder(agent: Agent, companies: List[Dict[str, str]], target_desc: str, offering_desc: str) -> List[Dict[str, Any]]:
    prompt = prompts.CONTACT_FINDER_PROMPT_TEMPLATE.format(
        target_desc=target_desc,
        offering_desc=offering_desc,
        companies_json=json.dumps(companies, ensure_ascii=False)
    )
    resp: RunOutput = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_research(agent: Agent, companies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompt = prompts.RESEARCH_PROMPT_TEMPLATE.format(
        companies_json=json.dumps(companies, ensure_ascii=False)
    )
    resp: RunOutput = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_email_writer(agent: Agent, contacts_data: List[Dict[str, Any]], research_data: List[Dict[str, Any]], offering_desc: str, sender_name: str, sender_company: str, calendar_link: Optional[str]) -> List[Dict[str, str]]:
    prompt = prompts.EMAIL_WRITER_PROMPT_TEMPLATE.format(
        sender_name=sender_name,
        sender_company=sender_company,
        offering_desc=offering_desc,
        calendar_link=calendar_link or 'N/A',
        contacts_json=json.dumps(contacts_data, ensure_ascii=False),
        research_json=json.dumps(research_data, ensure_ascii=False)
    )
    resp: RunOutput = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("emails", [])

def main() -> None:
    st.set_page_config(page_title="GTM B2B Outreach", layout="wide")

    # Sidebar: API keys
    st.sidebar.header("API Configuration")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    exa_key = st.sidebar.text_input("Exa API Key", type="password", value=os.getenv("EXA_API_KEY", ""))
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if exa_key:
        os.environ["EXA_API_KEY"] = exa_key

    if not openai_key or not exa_key:
        st.sidebar.warning("Enter both API keys to enable the app")

    # Inputs
    st.title("GTM B2B Outreach Multi Agent Team")
    st.info(
        "GTM teams often need to reach out for demos and discovery calls, but manual research and personalization is slow. "
        "This app uses GPT-5 with a multi-agent workflow to find target companies, identify contacts, research genuine insights (website + Reddit), "
        "and generate tailored outreach emails in your chosen style."
    )
    col1, col2 = st.columns(2)
    with col1:
        target_desc = st.text_area("Target companies (industry, size, region, tech, etc.)", height=100)
        offering_desc = st.text_area("Your product/service offering (1-3 sentences)", height=100)
    with col2:
        sender_name = st.text_input("Your name", value="Sales Team")
        sender_company = st.text_input("Your company", value="Our Company")
        calendar_link = st.text_input("Calendar link (optional)", value="")
        num_companies = st.number_input("Number of companies", min_value=1, max_value=10, value=5)
        email_style = st.selectbox(
            "Email style",
            options=["Professional", "Casual", "Cold", "Consultative"],
            index=0,
            help="Choose the tone/format for the generated emails",
        )

    if st.button("Start Outreach", type="primary"):
        # Validate
        if not openai_key or not exa_key:
            st.error("Please provide API keys in the sidebar")
        elif not target_desc or not offering_desc:
            st.error("Please fill in target companies and offering")
        else:
            # Stage-by-stage progress UI
            progress = st.progress(0)
            stage_msg = st.empty()
            details = st.empty()
            try:
                # Prepare agents
                company_agent = create_company_finder_agent()
                contact_agent = create_contact_finder_agent()
                research_agent = create_research_agent()
                email_agent = create_email_writer_agent(email_style)

                # 1. Companies
                stage_msg.info("1/4 Finding companies...")
                companies = run_company_finder(
                    company_agent,
                    target_desc.strip(),
                    offering_desc.strip(),
                    max_companies=int(num_companies),
                )
                progress.progress(25)
                details.write(f"Found {len(companies)} companies")

                # 2. Contacts
                stage_msg.info("2/4 Finding contacts (2–3 per company)...")
                contacts_data = run_contact_finder(
                    contact_agent,
                    companies,
                    target_desc.strip(),
                    offering_desc.strip(),
                ) if companies else []
                progress.progress(50)
                details.write(f"Collected contacts for {len(contacts_data)} companies")

                # 3. Research
                stage_msg.info("3/4 Researching insights (website + Reddit)...")
                research_data = run_research(research_agent, companies) if companies else []
                progress.progress(75)
                details.write(f"Compiled research for {len(research_data)} companies")

                # 4. Emails
                stage_msg.info("4/4 Writing personalized emails...")
                emails = run_email_writer(
                    email_agent,
                    contacts_data,
                    research_data,
                    offering_desc.strip(),
                    sender_name.strip() or "Sales Team",
                    sender_company.strip() or "Our Company",
                    calendar_link.strip() or None,
                ) if contacts_data else []
                progress.progress(100)
                details.write(f"Generated {len(emails)} emails")

                st.session_state["gtm_results"] = {
                    "companies": companies,
                    "contacts": contacts_data,
                    "research": research_data,
                    "emails": emails,
                }
                stage_msg.success("Completed")
            except Exception as e:
                stage_msg.error("Pipeline failed")
                st.error(f"{e}")

    # Show results if present
    results = st.session_state.get("gtm_results")
    if results:
        companies = results.get("companies", [])
        contacts = results.get("contacts", [])
        research = results.get("research", [])
        emails = results.get("emails", [])

        st.subheader("Top target companies")
        if companies:
            for idx, c in enumerate(companies, 1):
                st.markdown(f"**{idx}. {c.get('name','')}**  ")
                st.write(c.get("website", ""))
                st.write(c.get("why_fit", ""))
        else:
            st.info("No companies found")
        st.divider()

        st.subheader("Contacts found")
        if contacts:
            for c in contacts:
                st.markdown(f"**{c.get('name','')}**")
                for p in c.get("contacts", [])[:3]:
                    inferred = " (inferred)" if p.get("inferred") else ""
                    st.write(f"- {p.get('full_name','')} | {p.get('title','')} |  {p.get('email','')}{inferred}")
        else:
            st.info("No contacts found")
        st.divider()

        st.subheader("Research insights")
        if research:
            for r in research:
                st.markdown(f"**{r.get('name','')}**")
                for insight in r.get("insights", [])[:4]:
                    st.write(f"- {insight}")
        else:
            st.info("No research insights")
        st.divider()

        st.subheader("Suggested Outreach Emails")
        if emails:
            for i, e in enumerate(emails, 1):
                with st.expander(f"{i}. {e.get('company','')} → {e.get('contact','')}"):
                    st.write(f"Subject: {e.get('subject','')}")
                    st.text(e.get("body", ""))
        else:
            st.info("No emails generated")


if __name__ == "__main__":
    main()


