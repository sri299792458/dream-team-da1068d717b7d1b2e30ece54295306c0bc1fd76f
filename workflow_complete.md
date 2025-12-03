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

---

## 6. Agent Evolution & Knowledge Seeding

### Concept Initialization (Iteration 0)
**Location:** src/dream_team/evolution_agent.py (define_problem_space)

**Context Injected:**
- problem_statement
- target_metric
- column_schemas (Data structure information)
- techniques (Optional, if resuming)

**Description:** Extracts concepts from THREE sources to build the mathematical problem space:

1. **Domain Concepts (LLM):**
   - Uses prompt shown in Section 1 above
   - Extracts 10-20 concepts with importance (1-3) and category (domain/data/model/infra)
   - Source: llm

2. **Data Concepts (Schema Analysis - No LLM):**
   - Pure Python logic
   - If datetime columns  adds time_series_features (importance=2.0, category=DATA)
   - If categorical columns  adds categorical_encoding (importance=2.0, category=DATA)
   - Source: schema

3. **Model Concepts (Technique Analysis - No LLM):**
   - Converts technique list to concepts
   - Each technique  technique_name (importance=1.5, category=MODEL)
   - Source: code

**Output:**
- ProblemConcepts with normalized weights
- Stored in EvolutionAgent._concept_space
- Used to initialize agent depths (δ)

### Agent Depth Initialization
**Location:** src/dream_team/evolution_agent.py (initialize)

**Context Used:**
- Agent expertise text
- Agent title
- Concept names

**Logic (No LLM):**
`
for each agent:
  expertise_text = (agent.expertise + + agent.title).lower()
  for each concept:
    tokens = concept.replace(_,  ).split()
    if any(token in expertise_text for token in tokens):
      δ_0 = 0.8  # High initial depth
    else:
      δ_0 = 0.05  # Low initial depth
`

### New Agent Creation Flow

#### Step 1: Evolution Agent Proposes New Agents
**Location:** src/dream_team/evolution_agent.py (step)

**Mathematical Decision (No LLM):**

1. **Check if evolution needed:**
   - Requires 2 metric history points
   - Returns early if not enough data

2. **Update all agent depths (δ):**
   `
   quality = compute_quality()  # 0.8 if improved, 0.5 if plateau, 0.3 if worse
   for each agent:
     for each concept:
       learning = α  attention  quality  (1 - current_depth)
       forgetting = β  (1 - attention)  current_depth
       new_depth = current_depth + learning - forgetting
   `

3. **Compute coverage & gaps:**
   `
   coverage[concept] = max(agent.depths[concept] for all agents)
   gaps = [concept where coverage[concept] < gap_threshold (0.3)]
   `

4. **Propose deletion candidate:**
   - Only if team_size > min_size
   - Uses EvolutionPolicy.can_delete() to protect core agents
   - Scores based on: overlap + uniqueness + load
   - Weakest agent proposed for deletion

5. **Propose new agents (if team_size < max_size):**

   **a) Specialist from Generalist:**
   - Find agent with high overlap (>0.4) but low gini (<0.5)
   - Extract top 3 concepts with highest depths
   - Create NewAgentSpec:
     - kind = specialize
     - title = {Parent} Specialist
     - focus_concepts = [top 3 concepts]

   **b) Gap-Focused Specialists:**
   - For each gap concept, find best owner using EvolutionPolicy.score_gap_owner()
   - Group gaps by owner
   - Create specialists for each owner's gaps:
     - kind = specialize_gap
     - title = {Owner} ({concepts}) Specialist
     - focus_concepts = [assigned gaps]

   **c) Orphan Gap Agents:**
   - For concepts with no suitable owner (score < 0.2)
   - Create generic gap specialists:
     - kind = gap
     - title = Gap Specialist ({concepts})
     - expertise = Domain expert for: {concepts}

#### Step 2: Orchestrator Creates Agents
**Location:** src/dream_team/orchestrator.py (_evolve_team)

**Context Flow:**

For each NewAgentSpec:

1. **Create Agent object:**
   `python
   new_agent = Agent(
     title=spec.title,
     expertise=spec.expertise,
     goal=f optimize {target_metric},
     role=spec.role
   )
   `

2. **Seed Knowledge Base:**
   - Calls EvolutionAgent.seed_agent_knowledge()
   - See Knowledge Seeding section below

3. **Add to team:**
   - team_members.append(new_agent)
   - all_agents = [team_lead] + team_members

4. **Update Evolution Agent:**
   - Calls EvolutionAgent.update_team(all_agents)
   - Initializes math state for new agents (same logic as bootstrap)

### Knowledge Seeding
**Location:** src/dream_team/evolution_agent.py (seed_agent_knowledge)

**Context Injected:**
- focus_concepts (from NewAgentSpec)
- iteration_records (all past IterationRecords)

**Logic (No LLM):**

`python
for each iteration_record:
  for each technique in iteration_record.code_analysis.techniques:
    if technique matches any focus_concept:
      # Add to KB
      new_agent.knowledge_base.add_technique(technique)
      
      # If iteration was successful, add pattern
      if iteration had metrics:
        pattern = f Iter {i}: {technique} achieved {metric}={value}
        new_agent.knowledge_base.successful_patterns.append(pattern)
`

**Matching Logic:**
`python
technique_lower = technique.lower().replace(  , _)
for concept in focus_concepts:
  if (technique_lower in concept) or (concept in technique_lower):
    # Match found!
`

**Result:**
- New agent starts with relevant techniques mastered
- New agent starts with relevant success patterns
- New agent has NO error insights (fresh start)
- New agent has NO papers (starts empty)

### Agent Deletion Flow

**Step 1: Identify Deletion Candidate**
- TeamMathState.select_weakest_agent() computes deletion score
- Uses EvolutionPolicy.can_delete() to enforce constraints:
  - Cannot delete if role count < min_per_role (e.g., must have 1 lead, 1 coder)
  - Cannot delete if core=True (team lead, coding agent)

**Step 2: Remove from Team**
**Location:** src/dream_team/orchestrator.py (_evolve_team)

`python
if agent_to_remove in team_members:
  team_members.remove(agent_to_remove)
  # Knowledge base is LOST - not transferred
  # Math state will be removed in update_team()
`

**Step 3: Update Evolution State**
- EvolutionAgent.update_team() removes deleted agent from agent_states
- Concept coverage recalculated (may create new gaps)

### Concept Expansion (Dynamic)
**Location:** src/dream_team/evolution_agent.py (maybe_expand_concepts_from_techniques)

**Triggered:** After each iteration in orchestrator (_update_agent_knowledge_from_iteration)

**Context Injected:**
- techniques from CodeAnalysis

**Logic (No LLM):**

1. **Count technique occurrences:**
   - Techniques used 2 times candidates for concept expansion

2. **Filter new concepts:**
   - Skip if technique already a concept

3. **Add new concepts:**
   `python
   new_concept = Concept(
     name=technique_name,
     importance=0.5  max(existing_importances),
     category=MODEL,
     source=code_dynamic
   )
   `

4. **Re-normalize all concept weights**

5. **Initialize depths for all agents:**
   - Same token-matching logic as bootstrap
   - δ=0.3 if match, δ=0.05 otherwise

### Concept Refinement (Continuous)
**Location:** src/dream_team/evolution_agent.py (refine_concepts_from_code)

**Triggered:** After successful iteration in orchestrator (called after execution)

**Context Injected:**
- agent (coding agent who wrote code)
- techniques from CodeAnalysis

**Logic (No LLM):**

`python
for technique in techniques:
  for concept in agent.depths:
    if technique matches concept:
      # Boost depth (learning signal)
      agent.depths[concept] = min(1.0, current_depth + 0.1)
`

**Purpose:** Provide feedback from actual code execution to mathematical model

---

## 7. Bootstrap Recruitment Process

### Phase 1: PI Exploration
**Location:** src/dream_team/orchestrator.py (_bootstrap_exploration)

**Step 1: PI Decides Exploration Plan**

**Context Injected:**
- {problem_statement}
- {available_data} (list of variable names)

**Prompt:**
`
You've received a new research problem. Before assembling a team, you need to understand what you're dealing with.

## Problem:
{problem_statement}

## Available Data:
{list(executor.data_context.keys())}

## Your Task:
Decide what initial exploration will help you understand:
1. What the data looks like (schemas, sizes, distributions)
2. What the challenge involves
3. What expertise you'll need on your team

In 2-3 sentences, describe what exploration code should be written.
`

**Step 2: Coding Agent Implements Exploration**

**Context Injected:**
- {exploration_plan} (PI's request)
- {problem_statement}
- {available_variables}

**Prompt:**
`
The PI wants to do initial exploration. Write Python code to implement this:

## PI's Request:
{exploration_plan}

## Problem Statement (for reference):
{problem_statement}

## Available in execution context:
- Pre-imported libraries: pandas (pd), numpy (np), torch, pathlib.Path
- Variables: {list(executor.data_context.keys())}
  (You can use any of these variables directly in your code)

## Requirements:
- Inspect dataframes: print(df.info()), df.head(), df.describe(), df.columns
- ONLY print what you observe - no summaries, interpretations, or conclusions
- Use variables from Available in execution context above
- Suppress warnings if needed

Output ONLY the Python code, wrapped in `python blocks.
`

**Uses:** IndividualMeeting with use_react_coding=True (3-step planning)

**Step 3: Execute and Extract Schemas**

**Context Flow:**
1. Execute code with retry (up to 2 retries)
2. If successful:
   - Extract column schemas from dataframes
   - Store in self.column_schemas
   - Log execution event to EventStore

### Phase 2: PI Recruitment
**Location:** src/dream_team/orchestrator.py (_bootstrap_exploration continued)

**Context Injected:**
- {problem_statement}
- {exploration_results} (first 2000 chars of output)

**Prompt:**
`
Based on the problem and exploration results, decide what expertise you need on your team.

## Problem:
{problem_statement}

## Exploration Results:
{results['output'][:2000] if success else  Exploration failed, but you have the problem statement.}

## Your Task:
List 1-3 team members you want to recruit. For each, provide:
- Title (e.g., ML Strategist, Domain Expert, Data Analyst)
- Expertise (what they should know)
- Role (what they'll contribute)

Be specific about the skills needed based on what you learned from exploration and research papers.
Format your response as a simple list, one team member per line.
`

**Uses:** IndividualMeeting with optional paper search (if research_api available)

### Phase 3: Parse and Create Agents
**Location:** src/dream_team/orchestrator.py (_parse_and_recruit)

**Context Injected:**
- {recruitment_plan} (PI's text response)

**Prompt:**
`
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
  {
    title: Agent Title,
    expertise: Expertise description,
    role: Role description
  }
]
`

**Uses:** llm.generate_json()

**Agent Creation:**
`python
for each parsed_agent_spec:
  agent = Agent(
    title=spec[title],
    expertise=spec[expertise],
    goal= contribute specialized expertise to optimize the target metric,
    role=spec[role]
  )
  team_members.append(agent)
`

**Fallback:** If parsing fails, creates default ML Strategist

### Phase 4: Register with EvolutionPolicy
**Location:** src/dream_team/orchestrator.py (_initialize_evolution_agent)

**Context Used:**
- Agent expertise text

**Logic:**
`python
# Register PI
policy.register_agent(
  team_lead,
  AgentMeta(
    id=team_lead.title,
    role=lead,
    core=True,  # Protected from deletion
    tags=[leadership, planning, synthesis]
  )
)

# Register Coding Agent  
policy.register_agent(
  coding_agent,
  AgentMeta(
    id=coding_agent.title,
    role=coder,
    core=True,  # Protected from deletion
    tags=[python, implementation, debugging]
  )
)

# Register recruited members
for agent in team_members:
  tags = [t.strip() for t in agent.expertise.split(,)]
  policy.register_agent(
    agent,
    AgentMeta(
      id=agent.title,
      role=member,
      core=False,  # Can be deleted
      tags=tags
    )
  )
`

---

## 8. Knowledge Base Updates

### Update Triggers
**Location:** src/dream_team/orchestrator.py (main run loop)

**After each iteration:**
1. Semantic analysis creates IterationRecord
2. _update_agent_knowledge_from_iteration() called
3. All agents receive updates

### Update Logic
**Location:** src/dream_team/orchestrator.py (_update_agent_knowledge_from_iteration)

**Context Used:**
- iter_record.code_analysis.techniques
- iter_record.output_analysis.errors
- Current vs previous metric values

**For ALL agents:**

1. **Add Techniques:**
   `python
   for technique in code_analysis.techniques:
     agent.knowledge_base.add_technique(technique)
   `

2. **Add Success Pattern (if improvement):**
   `python
   if current_metric better than previous_metric:
     improvement = abs(current - previous)
     technique = techniques[0] if techniques else approach
     agent.knowledge_base.add_success_pattern(
       iteration=i,
       technique=technique,
       metric=target_metric,
       improvement=improvement
     )
   `

3. **Add Failure Pattern (if worse):**
   `python
   if current_metric worse than previous_metric:
     agent.knowledge_base.add_failure_pattern(
       iteration=i,
       technique=technique,
       metric=target_metric,
       reason=output_analysis.raw_summary
     )
   `

4. **Add Error Insights:**
   `python
   for error in output_analysis.errors:
     error_type = classify_error(error)  # NameError, KeyError, ValueError, etc.
     agent.knowledge_base.add_error_insight(
       iteration=i,
       error_type=error_type,
       error_msg=error[:200]
     )
   `

**Note:** ALL agents get the same updates - this is broadcast learning

---

## 9. State Restoration (Resume Flow)

### Event Replay
**Location:** src/dream_team/orchestrator.py (_restore_state)

**Phase 1: Load Events**
- Read events.json from disk
- Deserialize EventStore

**Phase 2: Restore Bootstrap (Iteration 0)**

1. **Restore Team:**
   - Load iteration_00_bootstrap.json
   - Recreate Agent objects from saved specs
   - Reconstruct team_members list

2. **Restore Variables:**
   - Find last successful bootstrap execution event
   - Extract code blob
   - Re-execute code in silence (executor.execute(code, silent=True))
   - Re-extract column schemas

**Phase 3: Restore Each Iteration**

For each iteration i > 0:

1. **Restore Execution State:**
   - Find last successful execution event
   - Extract code blob
   - Re-execute code silently
   - Variables now restored in executor

2. **Restore Team Composition:**
   - Find evolution events
   - Load new_team from evolution payload
   - Recreate Agent objects

3. **Restore Semantic State:**
   - Load meeting file for approach
   - Load execution blob for output
   - Load reflection event for reflection text
   - Re-run analyzers to recreate OutputAnalysis and CodeAnalysis
   - Construct IterationRecord

4. **Restore Evolution State:**
   - Call evolution_agent.refine_concepts_from_code()
   - Updates concept depths based on techniques

**Phase 4: Update Context Builder**
- context_builder.set_iterations(iteration_records)
- All past context now available for next iteration

**Knowledge Base Restoration:**
- Not explicitly restored in current implementation
- KBs start fresh and rebuild from iteration records
- **Potential Bug:** KB state lost on resume

---

## 10. Key Observations & Potential Issues

### 1. Hardcoded ReAct Steps
- Both _react_synthesis_task and _react_coding_task use hardcoded 3-step loops
- Step 0, 1, and 2 have specific, distinct prompts
- **Implication:** Cannot change max_steps without refactoring

### 2. Context Flow is Robust
-  **KB Injection:** kb_context injected into proposals
- **ContextBuilder:** Centralizes context creation effectively

### 3. Evolution Logic is Mostly Algorithmic
- Only one LLM call: initial concept extraction
- Rest is deterministic math (overlap, coverage, gaps, depths)
- Stable but potentially less creative in role invention

### 4. Reflection Loop Closes Learning
- What to avoid  KB pitfalls  future context
- Effective feedback loop from mistakes to learning

### 5. Knowledge Loss on Agent Deletion
- Deleted agent KBs not transferred
- **Potential Bug:** Valuable insights lost
- **Suggestion:** Redistribute KB before deletion

### 6. Knowledge Base Not Fully Restored on Resume
- Current _restore_state() doesn't explicitly restore KB
- **Potential Bug:** Subtle knowledge loss on resume
- **Suggestion:** Persist and restore KB state

### 7. All Agents Get Same KB Updates
- Broadcast learning: every agent learns from every iteration
- Good: Shared team knowledge
- Potential issue: Could dilute specialist focus

### 8. New Agent Seeding is Match-Based Only
- Knowledge seeding uses simple string matching
- May miss relevant but differently-named techniques
- **Suggestion:** Consider semantic similarity for matching

### 9. Concept Expansion Threshold
- Techniques need 2 uses to become concepts
- May be too conservative or too aggressive depending on domain
- **Suggestion:** Make threshold configurable

### 10. Evolution Decision Visibility
- Rich debug_info in EvolutionDecision
- Currently logged but not fed back to agents
- **Suggestion:** Could inform team about coverage gaps explicitly
