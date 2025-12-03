"""
Meeting module for Dream Team framework.

Implements team and individual meeting orchestration.
"""

from typing import List, Dict, Optional
from .agent import Agent
from .llm import get_llm
from datetime import datetime
import json
import os


class Meeting:
    """Base class for meetings"""

    def __init__(self, save_dir: Optional[str] = None, research_api=None):
        self.save_dir = save_dir
        self.transcript = []
        self.llm = get_llm()
        self.research_api = research_api  # Optional research API for on-demand paper search

    def add_message(self, agent_name: str, message: str):
        """Add message to transcript"""
        self.transcript.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "message": message
        })

    def save(self, filename: str):
        """Save meeting transcript"""
        if not self.save_dir:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.transcript, f, indent=2)

        print(f"💾 Meeting saved: {filepath}")


class TeamMeeting(Meeting):
    """Multi-agent team discussion"""

    def _react_synthesis_task(self, team_lead, agenda: str, context: str, temperature: float, is_final: bool, max_steps: int = 2) -> str:
        """
        ReAct loop for PI synthesis: iterative reasoning to synthesize team proposals.

        Simplified to 2 generic steps:
        1. Initial analysis - understand what was proposed
        2. Refinement - organize into coherent action plan

        This is internal reasoning only - no external search.
        """
        print(f"   🧠 {team_lead.title} using ReAct reasoning for synthesis...")

        reasoning_steps = []

        for step in range(max_steps):
            # Build context from previous thoughts
            previous_thoughts = ""
            if reasoning_steps:
                previous_thoughts = "\n\nPrevious reasoning:\n" + "\n".join([
                    f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)
                ])

            # Generic prompts that adapt to step number
            if step == 0:
                thinking_prompt = f"""You are synthesizing team proposals.

Agenda: {agenda}

Discussion so far:
{context}

Think step-by-step about the team's proposals:
- What are the key ideas and approaches suggested?
- Are there common themes or complementary ideas?
- What seems most promising for the goal?

Output only:
Thought: [Your analysis]
"""
            else:  # step >= 1
                thinking_prompt = f"""You are refining your synthesis.

Agenda: {agenda}

Discussion so far:
{context}
{previous_thoughts}

Refine your thinking:
- How should these ideas be organized into a coherent plan?
- What are the priorities and dependencies?
- What specific steps should the coding agent take?

Output only:
Thought: [Your refined thinking]
"""

            thought = self.llm.generate(
                thinking_prompt,
                system_instruction=team_lead.prompt,
                temperature=temperature * 0.7
            ).strip()

            # Remove "Thought:" prefix if present
            if thought.startswith('Thought:'):
                thought = thought.replace('Thought:', '').strip()

            print(f"      Step {step+1}: {thought[:100]}...")
            reasoning_steps.append(thought)

        # Final: Write synthesis based on all reasoning
        if is_final:
            final_prompt = f"""You are providing FINAL SYNTHESIS and DECISIONS.

Agenda: {agenda}

Discussion so far:
{context}

Your reasoning process:
{chr(10).join([f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)])}

Write your FINAL SYNTHESIS based on your reasoning.

Requirements:
- Make FINAL DECISIONS. Do not ask clarifying questions.
- Be specific: name the exact methods, features, or approaches to implement.
- Structure clearly for the coding agent.

Output 1-2 focused paragraphs.
"""
        else:
            final_prompt = f"""You are providing intermediate synthesis.

Agenda: {agenda}

Discussion so far:
{context}

Your reasoning process:
{chr(10).join([f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)])}

Now write your synthesis based on your reasoning.

As team lead, synthesize the key points and guide the next round of discussion.
Highlight areas of agreement and any gaps that need more exploration.

Keep it concise (1-2 paragraphs).
"""

        synthesis = self.llm.generate(
            final_prompt,
            system_instruction=team_lead.prompt,
            temperature=temperature * 0.8
        )

        return synthesis

    def run(
        self,
        team_lead: Agent,
        team_members: List[Agent],
        agenda: str,
        num_rounds: int = 2,
        temperature: float = 0.7
    ) -> Dict:
        """
        Run a team meeting

        Returns: {
            "summary": str,
            "decisions": List[str],
            "action_items": List[str]
        }
        """

        print(f"\n📋 TEAM MEETING")
        print(f"   Lead: {team_lead.title}")
        print(f"   Members: {[m.title for m in team_members]}")
        print(f"   Rounds: {num_rounds}\n")

        # Opening: Team lead sets context
        opening_prompt = f"""You are leading a team meeting.

Agenda: {agenda}

As the team lead, open the meeting by:
1. Framing the problem
2. Asking key questions for the team to address
3. Setting expectations for the discussion

Keep it concise (1-2 paragraphs).
"""

        opening = self.llm.generate(
            opening_prompt,
            system_instruction=team_lead.prompt,
            temperature=temperature
        )

        self.add_message(team_lead.title, opening)
        print(f"💬 {team_lead.title}:")
        print(f"{opening}\n")

        # Discussion rounds
        for round_num in range(num_rounds):
            print(f"--- Round {round_num + 1}/{num_rounds} ---\n")

            # Each member contributes
            for member in team_members:
                # Build context from transcript
                context = self._build_context()

                # Agent can optionally search papers if research_api is available
                if self.research_api:
                    response = self._optional_react_proposal(member, agenda, context, temperature)
                else:
                    # No research API available - simple proposal
                    
                    # Inject KB context
                    kb_data = member.knowledge_base.collect_for_intent("plan_next_iteration")
                    kb_context = ""
                    if kb_data["patterns"]:
                        kb_context += "\nYour Past Successes:\n" + "\n".join([f"- {p}" for p in kb_data["patterns"]])
                    if kb_data["techniques"]:
                        kb_context += "\nYour Mastered Techniques:\n" + "\n".join([f"- {t}" for t in kb_data["techniques"]])
                        
                    response = self.llm.generate(
                        f"""You are participating in a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

Relevant past learnings:
{kb_context}

Provide your input as {member.title}. Reference past learnings where directly applicable.
Keep it concise (1-2 paragraphs).
""",
                        system_instruction=member.prompt,
                        temperature=temperature
                    )

                self.add_message(member.title, response)
                member.meetings_participated += 1

                print(f"💬 {member.title}:")
                print(f"{response}\n")

            # Team lead synthesizes using ReAct
            is_final_round = (round_num == num_rounds - 1)
            context = self._build_context()

            synthesis = self._react_synthesis_task(
                team_lead=team_lead,
                agenda=agenda,
                context=context,
                temperature=temperature * 0.8,  # Slightly more focused
                is_final=is_final_round
            )

            self.add_message(team_lead.title, synthesis)
            print(f"💬 {team_lead.title} (synthesis):")
            print(f"{synthesis}\n")

        # Use the team lead's final synthesis as the summary
        # This is the actual action plan, not a generic overview
        # The last synthesis in the transcript is from the final round
        final_synthesis = synthesis  # This is the team lead's synthesis from the last round

        # Generate structured metadata from transcript
        metadata_prompt = f"""Extract structured metadata from this meeting transcript.

Transcript:
{self._build_context()}

Output valid JSON only, no other text:
{{
    "key_insights": ["insight 1", "insight 2"],
    "decisions": ["decision 1", "decision 2"],
    "action_items": ["action 1", "action 2"]
}}

Rules:
- All three fields are required.
- Use empty arrays [] if none found.
- No markdown, no ```json blocks.
"""

        metadata = self.llm.generate_json(metadata_prompt, temperature=0.3)

        # Return team lead's synthesis as the main summary
        return {
            "summary": final_synthesis,  # The detailed action plan
            "key_insights": metadata.get("key_insights", []),
            "decisions": metadata.get("decisions", []),
            "action_items": metadata.get("action_items", [])
        }

    def _build_context(self, max_messages: int = 10) -> str:
        """Build context string from recent transcript"""
        recent = self.transcript[-max_messages:]
        return "\n\n".join([
            f"{msg['agent']}: {msg['message']}"
            for msg in recent
        ])

    def _optional_react_proposal(self, agent, agenda: str, context: str, temperature: float, max_steps: int = 3) -> str:
        """
        Agent can OPTIONALLY search papers if they decide they need to.

        Pattern:
        1. Agent decides: Do I have enough knowledge or should I search?
        2. If search needed: Search papers and ground proposal
        3. Otherwise: Just provide proposal from expertise

        The agent has autonomy - they choose when to use the tool.
        """

        # Initial proposal with optional search
    
        # Inject KB context
        kb_data = agent.knowledge_base.collect_for_intent("plan_next_iteration")
        kb_context = ""
        if kb_data["patterns"]:
            kb_context += "\nYour Past Successes:\n" + "\n".join([f"- {p}" for p in kb_data["patterns"]])
        if kb_data["techniques"]:
            kb_context += "\nYour Mastered Techniques:\n" + "\n".join([f"- {t}" for t in kb_data["techniques"]])

        initial_prompt = f"""You are {agent.title} participating in a team meeting.

Agenda: {agenda}

Discussion so far:
{context}

Relevant past learnings:
{kb_context}

You have access to paper search. Use it if:
- The discussion mentions techniques you haven't implemented before
- You need recent benchmarks or SOTA results
- You want to cite supporting evidence

Otherwise, provide your proposal directly.

Tool format (if searching):
Thought: [Why searching helps]
Action: search_papers("[2-4 word query]")

If not searching, provide your proposal now (1-2 paragraphs).
"""

        response = self.llm.generate(
            initial_prompt,
            system_instruction=agent.prompt,
            temperature=temperature
        )

        # Check if agent wants to search
        react_history = []
        current_response = response

        for step in range(max_steps):
            # Parse response to see if agent is requesting search
            if 'search_papers(' in current_response.lower() or (
                'thought:' in current_response.lower() and
                'action:' in current_response.lower() and
                'search' in current_response.lower()
            ):
                # Agent wants to search - extract query
                thought = ""
                search_query = ""

                for line in current_response.split('\n'):
                    if line.strip().lower().startswith('thought:'):
                        thought = line.split(':', 1)[1].strip()
                    elif line.strip().lower().startswith('action:'):
                        action_text = line.split(':', 1)[1].strip()
                        # Extract query
                        if '"' in action_text:
                            search_query = action_text.split('"')[1]
                        elif "'" in action_text:
                            search_query = action_text.split("'")[1]
                        elif '(' in action_text and ')' in action_text:
                            # search_papers("query") format
                            query_part = action_text.split('(')[1].split(')')[0]
                            search_query = query_part.strip('"\'')

                if not search_query:
                    # Agent mentioned search but didn't format properly - use their response as-is
                    break

                print(f"      {agent.title} chose to search: '{search_query}'")

                # Execute search
                observation = self._search_and_observe(agent, search_query)

                react_history.append({
                    'thought': thought,
                    'action': f"search_papers('{search_query}')",
                    'observation': observation
                })

                # Ask agent for proposal now that they have papers
                follow_up_prompt = f"""You searched for papers and got results:

{observation}

Your reasoning so far:
{self._format_react_history(react_history)}

Now provide your final proposal based on your expertise and the papers you found.
Cite papers as supporting evidence: (Author et al., Year)

Keep it focused (1-2 paragraphs).
"""

                current_response = self.llm.generate(
                    follow_up_prompt,
                    system_instruction=agent.prompt,
                    temperature=temperature
                )
            else:
                # Agent provided direct proposal without searching - use it
                break

        return current_response



    def _format_react_history(self, history: list) -> str:
        """Format ReAct history for prompts"""
        if not history:
            return ""

        formatted = []
        for i, step in enumerate(history, 1):
            formatted.append(f"Step {i}:")
            formatted.append(f"  Thought: {step['thought']}")
            formatted.append(f"  Action: {step['action']}")
            formatted.append(f"  Observation: {step['observation']}")

        return '\n'.join(formatted)

    def _search_and_observe(self, agent, search_query: str) -> str:
        """Search papers and return observation summary with key insights"""
        try:
            # Limit query length
            if len(search_query) > 50:
                search_query = search_query[:50]

            # Search
            raw_results = self.research_api.search(
                query=search_query,
                limit=5,
                year_range=(2018, 2025)
            )

            if not raw_results:
                return "No relevant papers found."

            # Get existing papers
            existing_titles = [p.title for p in agent.knowledge_base.papers]

            # Add new papers with analysis
            papers_found = []
            for result in raw_results[:5]:
                if result.title not in existing_titles:
                    paper = result.to_paper()

                    # Analyze paper to extract key insights
                    # Fast analysis for iteration speed
                    analysis_prompt = f"""Extract 2-3 actionable insights from this paper.

Title: {paper.title}
Abstract: {paper.abstract}

Output a JSON array only, no other text:
["insight about method or technique", "insight 2", "insight 3"]

Focus on what can be directly applied to improve model performance."""

                    try:
                        insights = self.llm.generate_json(analysis_prompt, temperature=0.3)
                        if isinstance(insights, list):
                            paper.key_findings = insights[:3]
                    except Exception:
                        # Fallback: use first sentence of abstract
                        paper.key_findings = [paper.abstract.split('.')[0] + '.'] if paper.abstract else []

                    agent.knowledge_base.add_paper(paper)
                    papers_found.append(paper)
                    print(f"         ✓ {paper.title[:60]}... ({paper.year})")

            if not papers_found:
                return "Papers already in knowledge base."

            # Build observation summary with insights
            observation = f"Found {len(papers_found)} relevant papers:\n"
            observations = []
            for paper in papers_found:
                obs = f"- {paper.title[:60]}... ({', '.join(paper.authors[:2])} et al., {paper.year})"
                if paper.key_findings:
                    obs += f"\n  Key insights: {'; '.join(paper.key_findings[:2])}"
                observations.append(obs)

            observation += '\n'.join(observations)
            return observation

        except Exception as e:
            return f"Search failed: {e}"

class IndividualMeeting(Meeting):
    """One-on-one meeting with critic"""

    def _react_coding_task(self, agent, task: str, temperature: float, max_steps: int = 3) -> str:
        """
        ReAct loop for coding tasks: iterative reasoning to plan implementation.

        Pattern:
        1. Thought: Think about the approach/architecture
        2. Thought: Refine the approach and consider edge cases
        3. Thought: Finalize implementation details
        4. Final Answer: Write the complete code

        This is internal reasoning only - no external search.
        Used for coding agent to think through implementation step-by-step.
        """
        print(f"   🧠 {agent.title} using ReAct reasoning...")

        reasoning_steps = []

        for step in range(max_steps):
            # Build context from previous thoughts
            previous_thoughts = ""
            if reasoning_steps:
                previous_thoughts = "\n\nPrevious reasoning:\n" + "\n".join([
                    f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)
                ])

            # Iterative thinking prompts
            if step == 0:
                thinking_prompt = f"""You are planning how to implement a coding task.

Task: {task}

CONSTRAINT: All data variables already exist. You will not create or load any data.

Think step-by-step about the APPROACH:
- What's the overall architecture/structure?
- What are the main steps?
- What libraries/methods will you use?

Output only:
Thought: [Your thinking about the overall approach]
"""
            elif step == 1:
                thinking_prompt = f"""You are refining your implementation plan.

Task: {task}
{previous_thoughts}

Think step-by-step about IMPLEMENTATION DETAILS:
- What edge cases need handling?
- What's the data flow?
- What features/transformations are needed?

Output only:
Thought: [Your thinking about implementation details]
"""
            else:
                thinking_prompt = f"""You are finalizing your implementation plan.

Task: {task}
{previous_thoughts}

Think step-by-step about FINAL DETAILS:
- Are there any missing pieces?
- How will you ensure correctness?
- Any optimizations needed?

Output only:
Thought: [Your final thoughts before coding]
"""

            thought = self.llm.generate(
                thinking_prompt,
                system_instruction=agent.prompt,
                temperature=temperature * 0.7
            ).strip()

            # Remove "Thought:" prefix if present
            if thought.startswith('Thought:'):
                thought = thought.replace('Thought:', '').strip()

            print(f"      Step {step+1}: {thought[:100]}...")
            reasoning_steps.append(thought)

        # Final: Write code based on all reasoning
        final_prompt = f"""You are implementing a coding task.

Task: {task}

Your reasoning process:
{...}

Write COMPLETE, EXECUTABLE Python code based on your reasoning.

Constraints:
- All data variables already exist. Do NOT create, overwrite, or mock them.
- Use exact column names from the schema provided.
- Import all required libraries at the top.

Output ONLY the code in ```python blocks.
"""


        final_output = self.llm.generate(
            final_prompt,
            system_instruction=agent.prompt,
            temperature=temperature * 0.9
        )

        return final_output

    def _optional_search_task(self, agent, task: str, temperature: float, max_steps: int = 3) -> str:
        """
        Agent can OPTIONALLY search papers if they decide they need to.

        Pattern:
        1. Agent decides: Do I have enough knowledge or should I search?
        2. If search needed: Search papers and use insights
        3. Otherwise: Just complete task from expertise

        The agent has autonomy - they choose when to use the tool.
        """

        # Initial task with optional search
        initial_prompt = f"""Task: {task}

You have access to paper search. Use it if:
- You need recent research or benchmarks
- You want to cite supporting evidence
- The task requires external knowledge

Otherwise, complete the task directly.

Tool format (if searching):
Thought: [Why searching helps]
Action: search_papers("[2-4 word query]")

If not searching, complete the task now.
"""

        response = self.llm.generate(
            initial_prompt,
            system_instruction=agent.prompt,
            temperature=temperature
        )

        # Check if agent wants to search
        react_history = []
        current_response = response

        for step in range(max_steps):
            # Parse response to see if agent is requesting search
            if 'search_papers(' in current_response.lower() or (
                'thought:' in current_response.lower() and
                'action:' in current_response.lower() and
                'search' in current_response.lower()
            ):
                # Agent wants to search - extract query
                thought = ""
                search_query = ""

                for line in current_response.split('\n'):
                    if line.strip().lower().startswith('thought:'):
                        thought = line.split(':', 1)[1].strip()
                    elif line.strip().lower().startswith('action:'):
                        action_text = line.split(':', 1)[1].strip()
                        # Extract query
                        if '"' in action_text:
                            search_query = action_text.split('"')[1]
                        elif "'" in action_text:
                            search_query = action_text.split("'")[1]
                        elif '(' in action_text and ')' in action_text:
                            query_part = action_text.split('(')[1].split(')')[0]
                            search_query = query_part.strip('"\'')

                if not search_query:
                    # Agent mentioned search but didn't format properly - use their response as-is
                    break

                print(f"      {agent.title} chose to search: '{search_query}'")

                # Execute search
                observation = self._search_and_observe(agent, search_query)

                react_history.append({
                    'thought': thought,
                    'action': f"search_papers('{search_query}')",
                    'observation': observation
                })

                # Ask agent to complete task with papers
                follow_up_prompt = f"""You searched for papers and got results:

{observation}

Your reasoning so far:
{self._format_react_history(react_history)}

Now complete the task based on your expertise and the papers you found.
Cite papers as supporting evidence: (Author et al., Year)

Be specific, detailed, and actionable.
"""

                current_response = self.llm.generate(
                    follow_up_prompt,
                    system_instruction=agent.prompt,
                    temperature=temperature
                )
            else:
                # Agent completed task without searching - use it
                break

        return current_response

    def _react_individual_task(self, agent, task: str, temperature: float, max_steps: int = 2) -> str:
        """
        ReAct loop for individual task: agent reasons and searches papers before final output.

        Pattern:
        1. Thought: Think about the task and what approach to take
        2. Action: Search papers to inform the decision
        3. Observation: Papers found
        4. (Repeat to build grounded thinking)
        5. Final Answer: Complete the task with citations

        Used for bootstrap recruitment and other individual tasks.
        """
        print(f"   🧠 {agent.title} using ReAct reasoning...")

        react_history = []

        for step in range(max_steps):
            # Thought: Think about task and what to search
            thought_prompt = f"""You are working on a task.

Task: {task}

{"Previous reasoning:" if react_history else ""}
{self._format_react_history(react_history)}

Based on YOUR EXPERTISE, think about how to approach this task.
What do you need to know? What should you search for to inform your decision?

Output format:
Thought: [Your thinking about how to approach this task]
Action: Search papers on "[2-4 word search query]" to inform this decision

Be concise. Only output Thought and Action.
"""

            thought_action = self.llm.generate(
                thought_prompt,
                system_instruction=agent.prompt,
                temperature=temperature * 0.8
            )

            # Parse thought and action
            thought = ""
            search_query = ""

            for line in thought_action.split('\n'):
                if line.startswith('Thought:'):
                    thought = line.replace('Thought:', '').strip()
                elif line.startswith('Action:'):
                    action_text = line.replace('Action:', '').strip()
                    # Extract query from "Search papers on 'X'" or similar
                    if '"' in action_text:
                        search_query = action_text.split('"')[1]
                    elif "'" in action_text:
                        search_query = action_text.split("'")[1]
                    else:
                        # Fallback: use last few words
                        words = action_text.split()
                        search_query = ' '.join(words[-4:]) if len(words) > 4 else action_text

            if not search_query:
                break  # Stop if can't parse

            print(f"      Step {step+1} Thought: {thought[:80]}...")
            print(f"      Step {step+1} Action: Search '{search_query}'")

            # Action: Search papers
            observation = self._search_and_observe(agent, search_query)

            print(f"      Step {step+1} Observation: {observation[:100]}...")

            react_history.append({
                'thought': thought,
                'action': f"Search papers on '{search_query}'",
                'observation': observation
            })

        # Final Answer: Complete task with citations
        final_prompt = f"""You are completing a task.

Task: {task}

Your ReAct reasoning process:
{self._format_react_history(react_history)}

Complete the task based on YOUR EXPERTISE and the research you've done.
Use papers you found as SUPPORTING EVIDENCE to inform your decisions.

**Cite relevant papers to support your recommendations.**
Format citations as: (Author et al., Year)

Be specific, detailed, and actionable.
"""

        final_output = self.llm.generate(
            final_prompt,
            system_instruction=agent.prompt,
            temperature=temperature * 0.9
        )

        return final_output

    def _format_react_history(self, history: list) -> str:
        """Format ReAct history for prompts"""
        if not history:
            return ""

        formatted = []
        for i, step in enumerate(history, 1):
            formatted.append(f"Step {i}:")
            formatted.append(f"  Thought: {step['thought']}")
            formatted.append(f"  Action: {step['action']}")
            formatted.append(f"  Observation: {step['observation']}")

        return '\n'.join(formatted)

    def _search_and_observe(self, agent, search_query: str) -> str:
        """Search papers and return observation summary with key insights"""
        try:
            # Limit query length
            if len(search_query) > 50:
                search_query = search_query[:50]

            # Search
            raw_results = self.research_api.search(
                query=search_query,
                limit=5,
                year_range=(2018, 2025)
            )

            if not raw_results:
                return "No relevant papers found."

            # Get existing papers
            existing_titles = [p.title for p in agent.knowledge_base.papers]

            # Add new papers with analysis
            papers_found = []
            for result in raw_results[:5]:
                if result.title not in existing_titles:
                    paper = result.to_paper()

                    # Analyze paper to extract key insights
                    analysis_prompt = f"""Extract 2-3 key actionable insights from this paper abstract.

Title: {paper.title}
Abstract: {paper.abstract}

Output ONLY a JSON array of 2-3 brief insights:
["insight 1", "insight 2", "insight 3"]

Focus on methods, findings, or techniques that could be applied."""

                    try:
                        insights = self.llm.generate_json(analysis_prompt, temperature=0.3)
                        if isinstance(insights, list):
                            paper.key_findings = insights[:3]
                    except Exception:
                        # Fallback: use first sentence of abstract
                        paper.key_findings = [paper.abstract.split('.')[0] + '.'] if paper.abstract else []

                    agent.knowledge_base.add_paper(paper)
                    papers_found.append(paper)
                    print(f"         ✓ {paper.title[:60]}... ({paper.year})")

            if not papers_found:
                return "Papers already in knowledge base."

            # Build observation summary with insights
            observation = f"Found {len(papers_found)} relevant papers:\n"
            observations = []
            for paper in papers_found:
                obs = f"- {paper.title[:60]}... ({', '.join(paper.authors[:2])} et al., {paper.year})"
                if paper.key_findings:
                    obs += f"\n  Key insights: {'; '.join(paper.key_findings[:2])}"
                observations.append(obs)

            observation += '\n'.join(observations)
            return observation

        except Exception as e:
            return f"Search failed: {e}"

    def run(
        self,
        agent: Agent,
        task: str,
        critic_agent: Optional[Agent] = None,
        num_iterations: int = 2,
        temperature: float = 0.7,
        use_react: bool = False,
        use_react_coding: bool = False
    ) -> str:
        """
        Run individual meeting with iterative refinement

        Args:
            use_react: If True and research_api available, use ReAct to search papers
            use_react_coding: If True, use ReAct for coding (internal reasoning, no search)

        Returns: Final output
        """

        print(f"\n👤 INDIVIDUAL MEETING")
        print(f"   Agent: {agent.title}")
        print(f"   Iterations: {num_iterations}")
        if use_react and self.research_api:
            print(f"   Using ReAct: Yes (with paper search)")
        elif use_react_coding:
            print(f"   Using ReAct: Yes (internal reasoning)")
        print()

        # Initial work - choose approach
        if use_react_coding:
            # Coding agent: iterative reasoning without external search
            output = self._react_coding_task(agent, task, temperature)
        elif use_react and self.research_api:
            # FORCED ReAct with paper search (legacy - forces search)
            output = self._react_individual_task(agent, task, temperature)
        elif self.research_api:
            # research_api available - agent can OPTIONALLY search
            output = self._optional_search_task(agent, task, temperature)
        else:
            # No research API - simple generation
            work_prompt = f"""Task: {task}

Complete this task drawing on your expertise and knowledge base.
Be specific, detailed, and actionable.
"""

            output = self.llm.generate(
                work_prompt,
                system_instruction=agent.prompt,
                temperature=temperature
            )

        self.add_message(agent.title, output)
        agent.meetings_participated += 1

        print(f"💬 {agent.title} (initial):")
        print(f"{output[:200]}...\n")

        # Iterative refinement with critic
        if critic_agent:
            for iteration in range(num_iterations):
                print(f"--- Iteration {iteration + 1}/{num_iterations} ---\n")

                # Critic provides feedback
                critique_prompt = f"""You are reviewing work from {agent.title}.

Task: {task}

Their output:
{output}

Provide constructive criticism:
1. What's done well?
2. What's missing or could be improved?
3. Specific suggestions for refinement

Be brief but specific.
"""

                critique = self.llm.generate(
                    critique_prompt,
                    system_instruction=critic_agent.prompt,
                    temperature=temperature * 0.8
                )

                self.add_message(critic_agent.title, critique)
                print(f"💬 {critic_agent.title}:")
                print(f"{critique}\n")

                # Agent revises
                revision_prompt = f"""Task: {task}

Your previous output:
{output}

Feedback from {critic_agent.title}:
{critique}

Revise your work based on this feedback. Improve and extend as needed.
"""

                output = self.llm.generate(
                    revision_prompt,
                    system_instruction=agent.prompt,
                    temperature=temperature
                )

                self.add_message(agent.title, output)
                print(f"💬 {agent.title} (revised):")
                print(f"{output[:200]}...\n")

        return output
