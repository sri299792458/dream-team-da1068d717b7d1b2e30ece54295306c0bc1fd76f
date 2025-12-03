Dream Team Prompt Flow & Context Report
This report provides a comprehensive, fact-checked analysis of every prompt used in the Dream Team framework, detailing the exact context injected at each step. It is based on a code audit of 
orchestrator.py
, 
meetings.py
, 
agent.py
, 
context_builder.py
, 
evolution_agent.py
, and 
analyzers.py
.

1. System Initialization & Evolution
Agent System Prompts
Location: 
src/dream_team/agent.py
 (Agent.prompt) Context Injected: None (Static based on agent attributes) Description: Defines the agent's persona.

Base Prompt:

You are {self.title}.
Expertise: {self.expertise}
Goal: {self.goal}
Role: {self.role}
{closing}
Closing (Lead):

You are part of a research team solving data science challenges.
As the lead, ensure your approach embodies rigorous experimental methodology. Consider what constitutes sound scientific practice: how to validate hypotheses, establish baselines, make incremental progress, and learn systematically from results. Apply your full expertise to guide the team toward methodologically sound decisions.
Closing (Member):

You are part of a research team solving data science challenges.
When proposing approaches, consider both domain expertise and experimental rigor. Think about validation, incremental progress, and learning from previous work. Provide insightful, actionable contributions grounded in sound methodology.
Evolution Concept Extraction
Location: 
src/dream_team/evolution_agent.py
 (
_build_domain_concepts_from_llm
) Context Injected:

{problem_statement}
{target_metric} Description: Extracts 10-20 key technical concepts to track for evolution.
Prompt:

You are helping configure a multi-agent research team for this problem.
Problem:
{problem_statement}
Target metric: {target_metric}
Task:
Extract the 10-20 most important technical concepts, domain ideas, or methods
that the team should explicitly track.
Rules:
- Output ONE concept per line.
- Format: concept_name | importance | category
- concept_name: short, snake_case, no spaces (e.g. gradient_boosting, microbial_growth)
- importance: integer 1, 2, or 3 (3 = very important).
- category: one of {{domain, data, model, infra}}. If unsure, use domain.
- No explanations, no headings, only the raw list.
2. Bootstrap Phase (Iteration 0)
Exploration Task
Location: 
src/dream_team/orchestrator.py
 (
_bootstrap_exploration
) Context Injected:

{problem_statement}
{list(self.executor.data_context.keys())} (Available variables) Description: Asks the PI to decide what initial data exploration is needed.
Prompt:

You've received a new research problem. Before assembling a team, you need to understand what you're dealing with.
## Problem:
{problem_statement}
## Available Data:
{list(self.executor.data_context.keys())}
## Your Task:
Decide what initial exploration will help you understand:
1. What the data looks like (schemas, sizes, distributions)
2. What the challenge involves
3. What expertise you'll need on your team
In 2-3 sentences, describe what exploration code should be written.
Recruitment Task
Location: 
src/dream_team/orchestrator.py
 (
_bootstrap_exploration
) Context Injected:

{problem_statement}
{results['output']} (Output from exploration code) Description: Asks the PI to recruit team members based on exploration results.
Prompt:

Based on the problem and exploration results, decide what expertise you need on your team.
## Problem:
{problem_statement}
## Exploration Results:
{results['output'][:2000] if results['success'] else "Exploration failed, but you have the problem statement."}
## Your Task:
List 1-3 team members you want to recruit. For each, provide:
- Title (e.g., "ML Strategist", "Domain Expert", "Data Analyst")
- Expertise (what they should know)
- Role (what they'll contribute)
Be specific about the skills needed based on what you learned from exploration and research papers.
Format your response as a simple list, one team member per line.
Recruitment Parsing
Location: 
src/dream_team/orchestrator.py
 (
_parse_and_recruit
) Context Injected: {recruitment_plan} Description: Extracts structured JSON agent specs from the PI's text plan.

Prompt:

Parse this recruitment plan and extract agent specifications.
## Recruitment Plan:
{recruitment_plan}
## Your Task:
Extract the team members mentioned in the plan.
For each, provide:
- Title
- Expertise
- Role
Output ONLY a JSON list of objects, like this:
[
  {{
    "title": "Agent Title",
    "expertise": "Expertise description",
    "role": "Role description"
  }}
]
3. Planning Phase (Team Meeting)
Meeting Opening
Location: 
src/dream_team/meetings.py
 (TeamMeeting.run) Context Injected: {agenda} Description: Team Lead frames the problem and sets expectations.

Prompt:

You are leading a team meeting.
Agenda: {agenda}
As the team lead, open the meeting by:
1. Framing the problem
2. Asking key questions for the team to address
3. Setting expectations for the discussion
Keep it concise (1-2 paragraphs).
Member Proposal (Standard)
Location: 
src/dream_team/meetings.py
 (TeamMeeting.run) Context Injected:

{agenda}
{context} (Recent meeting transcript)
{kb_context} (CRITICAL: Injected from 
KnowledgeBase
)
"Your Past Successes: ..."
"Your Mastered Techniques: ..." Description: Asks a team member for their input, explicitly showing them their past successes.
Prompt:

You are participating in a team meeting.
Agenda: {agenda}
Discussion so far:
{context}
{kb_context}
Provide your input as {member.title}. Draw on your expertise and past learnings.
Keep it concise (1-2 paragraphs).
Member Proposal (with Research Tool)
Location: 
src/dream_team/meetings.py
 (
_optional_react_proposal
) Context Injected:

{agenda}
{context}
{kb_context} (Same as above) Description: Gives the agent the option to search papers or answer directly.
Initial Prompt:

You are {agent.title} participating in a team meeting.
Agenda: {agenda}
Discussion so far:
{context}
{kb_context}
You have access to a paper search tool if you need it. Based on the problem:
- If you already have sufficient expertise and knowledge, just provide your proposal directly
- If you need to verify something or find supporting evidence, you can search papers
**Available Tool**:
- search_papers(query): Search Semantic Scholar for recent papers (2018-2025)
Provide your input as {agent.title}. You may:
1. Just give your proposal if you have enough knowledge
2. Or use the tool format if you want to search:
   Thought: [Why I need to search]
   Action: search_papers("[your query]")
Then wait for results before giving your final proposal.
If you don't need to search, just provide your proposal directly (1-2 paragraphs).
Follow-up Prompt (if searched):

You searched for papers and got results:
{observation}
Your reasoning so far:
{self._format_react_history(react_history)}
Now provide your final proposal based on your expertise and the papers you found.
Cite papers as supporting evidence: (Author et al., Year)
Keep it focused (1-2 paragraphs).
Team Lead Synthesis (ReAct)
Location: 
src/dream_team/meetings.py
 (
_react_synthesis_task
) Context Injected: {agenda}, {context}, {previous_thoughts} Description: Hardcoded 3-Step Reasoning Process

Step 0 (Analysis):

You are synthesizing team proposals.
Agenda: {agenda}
Discussion so far:
{context}
Think step-by-step about WHAT WAS PROPOSED:
- What did each team member propose?
- What are the key ideas?
- Are there common themes?
Output only:
Thought: [Your analysis of team proposals]
Step 1 (Priorities):

You are refining your synthesis.
Agenda: {agenda}
Discussion so far:
{context}
{previous_thoughts}
Think step-by-step about PRIORITIES AND DEPENDENCIES:
- Are there any conflicts between proposals?
- What needs to be done first?
- What's most important for the goal?
Output only:
Thought: [Your thinking about priorities and dependencies]
Step 2 (Action Plan):

You are finalizing your synthesis.
Agenda: {agenda}
Discussion so far:
{context}
{previous_thoughts}
Think step-by-step about the ACTION PLAN:
- What specific steps should the coding agent take?
- In what order?
- Any important details to emphasize?
Output only:
Thought: [Your final thoughts on the action plan]
Final Synthesis:

You are providing FINAL SYNTHESIS and DECISIONS.
Agenda: {agenda}
Discussion so far:
{context}
Your reasoning process:
{chr(10).join([f"Step {i+1}: {thought}" for i, thought in enumerate(reasoning_steps)])}
Now write your FINAL SYNTHESIS based on your reasoning.
IMPORTANT:
- Make FINAL DECISIONS, do NOT ask clarifying questions
- Synthesize what the team proposed into a clear action plan
- Be decisive and specific about what to implement
- Structure it clearly for the coding agent to understand
Keep it focused (1-2 paragraphs).
4. Implementation Phase
Coding Task
Location: 
src/dream_team/orchestrator.py
 (
_implement_approach
) Context Injected: {impl_context} (Built by ContextBuilder.for_coding)

Data Schema: {column_schemas}
Team Plan: {team_plan}
Last Iteration Insights: Key observations & errors from previous run.
Proven Techniques: From 
KnowledgeBase
 (techniques & patterns). Description: The primary prompt for the Coding Agent.
Prompt:

Implement the team's plan.
{impl_context}
## EXECUTION ENVIRONMENT:
- The following variables are PRE-LOADED in the global scope: {list(self.executor.data_context.keys())}
- You MUST use these existing variables.
- **CRITICAL: DO NOT define any of the variables listed above. They already exist.**
- **CRITICAL: DO NOT generate dummy data. It will overwrite the real data.**
- Just start using the variables directly.
## Requirements:
- **CRITICAL: Check "Data Schema" above and use EXACT column names. Do not hallucinate columns.**
- Use GPU when training models
- Import what you need (standard libraries)
- Compute {self.target_metric} and store in a variable if training a model
- Print important outputs: metrics, feature importance, model summaries
- Save trained models (joblib.dump, torch.save) if training is expensive
- Suppress verbose output (warnings.filterwarnings('ignore'), verbose=0/-1)
Output ONLY Python code in ```python``` blocks.
Individual Coding ReAct
Location: 
src/dream_team/meetings.py
 (
_react_coding_task
) Context Injected: {task}, {previous_thoughts} Description: Hardcoded 3-Step Coding Thought Process

Step 0 (Approach):

You are planning how to implement a coding task.
Task: {task}
Think step-by-step about the APPROACH:
- What's the overall architecture/structure?
- What are the main steps?
- What libraries/methods will you use?
Output only:
Thought: [Your thinking about the overall approach]
Step 1 (Details):

You are refining your implementation plan.
Task: {task}
{previous_thoughts}
Think step-by-step about IMPLEMENTATION DETAILS:
- What edge cases need handling?
- What's the data flow?
- What features/transformations are needed?
Output only:
Thought: [Your thinking about implementation details]
Step 2 (Finalization):

You are finalizing your implementation plan.
Task: {task}
{previous_thoughts}
Think step-by-step about FINAL DETAILS:
- Are there any missing pieces?
- How will you ensure correctness?
- Any optimizations needed?
Output only:
Thought: [Your final thoughts before coding]
Final Code Generation:

You are implementing a coding task.
Task: {task}
Your reasoning process:
{...}
Now write the COMPLETE, EXECUTABLE code based on your reasoning.
Important constraints:
- Do NOT create or overwrite any preloaded data variables; assume they already exist.
- Do NOT construct dummy toy datasets for these variables.
Output ONLY the code in ```python blocks.
Error Fixing
Location: 
src/dream_team/orchestrator.py
 (via 
_execute_with_retry
 -> ContextBuilder.for_error_fix) Context Injected:

Error Details: Error message & traceback.
Original Approach: What was attempted.
Similar Past Errors: From ReflectionMemory.
Recent Error Patterns: From last iteration analysis.
Known Pitfalls: From 
KnowledgeBase
. Description: Context provided when code execution fails.
5. Analysis & Reflection Phase
Output Analysis
Location: 
src/dream_team/analyzers.py
 (
OutputAnalyzer
) Context Injected: {output}, {error}, {traceback} Description: Extracts semantic meaning from raw logs.

Prompt:

Analyze this code execution output and extract semantic information.
**Output:**
{output}

**Error:** {error} (if present)
**Traceback:** {traceback} (if present)
Return a JSON object with:
{
  "summary": "Brief summary of what happened",
  "errors": ["error message 1", "error message 2"],
  "warnings": ["warning 1", "warning 2"],
  "metrics": {"metric_name": value},
  "observations": ["key observation 1", "key observation 2"]
}
For observations, identify issues like:
- Overfitting (train vs validation gap)
- Data leakage
- Convergence problems
- Class imbalance issues
- Memory issues
- Performance bottlenecks
Only include observations that are clearly evident in the output.
Code Analysis
Location: 
src/dream_team/analyzers.py
 (
CodeAnalyzer
) Context Injected: {code} Description: Understands what the code actually did.

Prompt:

Analyze this Python code and extract key information.
**Code:**
```python
{code[:8000]}
Return a JSON object with: {{ "techniques": ["technique1", "technique2"], "key_decisions": ["decision1", "decision2"], "libraries": ["library1", "library2"], "complexity": "simple|moderate|complex", "data_flow": "Brief description of data flow" }}

For techniques, identify ML/data science techniques like:

gradient_boosting, random_forest, neural_network
cross_validation, grid_search, feature_engineering
regularization, ensemble, stacking
dimensionality_reduction, clustering
For key_decisions, identify important design choices:

Model selection rationale
Feature engineering approach
Validation strategy
Hyperparameter tuning method
For complexity:

simple: Basic model, straightforward preprocessing
moderate: Multiple models or advanced preprocessing
complex: Ensemble, extensive feature engineering, or custom architectures
For data_flow:

Briefly describe how data moves through the code (load → preprocess → train → evaluate)
### Reflection
**Location:** [src/dream_team/orchestrator.py](file:///c:/Users/srini/Downloads/dream-team-da1068d717b7d1b2e30ece54295306c0bc1fd76f/src/dream_team/orchestrator.py) ([_reflect_on_iteration](file:///c:/Users/srini/Downloads/dream-team-da1068d717b7d1b2e30ece54295306c0bc1fd76f/src/dream_team/orchestrator.py#1216-1307))
**Context Injected:**
- `{approach}`
- `{code}` (snippet)
- `{results}` (success/fail)
- `{metrics}`
- `{output_preview}`
- `{history_section}` (Previous failed attempts in this iteration)
**Description:** The "Reflexion" step where the Team Lead learns.
**Prompt:**
```text
You are reviewing iteration {self.iteration} of this research experiment.
## What Was Attempted
{approach}
## Code Implementation
```python
{code[:500]}...
What Happened
Success: {results.get('success')} Metrics: {metrics}

Execution Output
{output_preview} {error_section} {history_section}

Reflection Task
Analyze this iteration and extract concrete, actionable learnings.

Answer these questions:

What worked? - Specific techniques, patterns, or approaches that succeeded
What failed? - Specific mistakes, wrong assumptions, or bugs encountered
Why did it fail? - Root cause analysis (not just symptoms)
What to try differently? - Concrete suggestions for the next iteration
What to avoid? - Dead ends or approaches that won't work for this problem
Be specific and actionable. Focus on insights the team can actually use.

Format your response with these exact section headers: Worked: [what succeeded] Failed: [what didn't work] Why: [root cause] Try next: [specific suggestions] Avoid: [what not to do]

---
## Key Observations & Potential Issues
1.  **Hardcoded ReAct Steps:**
    *   Both [_react_synthesis_task](file:///c:/Users/srini/Downloads/dream-team-da1068d717b7d1b2e30ece54295306c0bc1fd76f/src/dream_team/meetings.py#49-187) (Team Lead) and [_react_coding_task](file:///c:/Users/srini/Downloads/dream-team-da1068d717b7d1b2e30ece54295306c0bc1fd76f/src/dream_team/meetings.py#542-648) (Coding Agent) use a **hardcoded 3-step loop**.
    *   Step 0, 1, and 2 have specific, distinct prompts.
    *   **Implication:** You cannot simply change `max_steps` to 2 without breaking the logic, as it would skip the "Finalization/Action Plan" thought step.
2.  **Context Flow is Robust:**
    *   **KB Injection:** We successfully verified that `kb_context` (Past Successes/Techniques) is now injected into both standard and ReAct-based member proposals.
    *   **ContextBuilder:** The [ContextBuilder](file:///c:/Users/srini/Downloads/dream-team-da1068d717b7d1b2e30ece54295306c0bc1fd76f/src/dream_team/context_builder.py#14-323) centralizes context creation effectively, ensuring that `impl_context` (for coding) and `base_context` (for planning) consistently include relevant history, metrics, and schemas.
3.  **Evolution Logic is mostly Algorithmic:**
    *   The [EvolutionAgent](file:///c:/Users/srini/Downloads/dream-team-da1068d717b7d1b2e30ece54295306c0bc1fd76f/src/dream_team/evolution_agent.py#388-1175) relies heavily on Python logic (math state, overlap calculation) rather than LLM prompts for its decisions. The only LLM call is for initial concept extraction. This makes it deterministic and stable, but potentially less "creative" in inventing new roles.
4.  **Reflection Loop:**
    *   The reflection prompt is quite structured and explicitly asks for "What to avoid", which feeds directly into the [KnowledgeBase](file:///c:/Users/srini/Downloads/dream-team-da1068d717b7d1b2e30ece54295306c0bc1fd76f/src/dream_team/agent.py#43-134) "pitfalls" for future iterations. This closes the learning loop effectively.