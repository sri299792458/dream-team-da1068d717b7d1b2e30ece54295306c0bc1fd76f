# Dream Team Prompt Report

This document contains a comprehensive list of all prompts used in the Dream Team framework, extracted from the codebase.

## 1. Agent System Prompts (`src/dream_team/agent.py`)

### `Agent.prompt` property
The base system prompt for all agents.

```text
You are {self.title}.

## Your Profile
- **Expertise**: {self.expertise}
- **Role**: {self.role}
- **Goal**: {self.goal}

## Operational Rules
1. You are part of an autonomous AI research team.
2. Your goal is to help the team achieve the target metric.
3. Be concise, objective, and constructive.
4. Focus on your specific area of expertise.
5. If you don't know something, admit it.

{role_specific_closing}
```

## 2. Meeting Prompts (`src/dream_team/meetings.py`)

### `TeamMeeting.run`
**Opening Prompt (Team Lead):**
```text
You are leading a team meeting.

Agenda: {agenda}

As the team lead, open the meeting by:
1. Framing the problem
2. Asking key questions for the team to address
3. Setting expectations for the discussion

Keep it concise (1-2 paragraphs).
```

**Member Response Prompt:**
```text
You are participating in a team meeting.

Agenda: {agenda}

Discussion so far:
{context}
{kb_context}

Provide your input as {member.title}. Draw on your expertise and past learnings.
Keep it concise (1-2 paragraphs).
```

### `TeamMeeting._react_synthesis_task` (ReAct Loop)
**Step 0 (Initial Analysis):**
```text
You are synthesizing team proposals.

Agenda: {agenda}

Discussion so far:
{context}

Think step-by-step about the team's proposals:
- What are the key ideas and approaches suggested?
- Are there common themes or complementary ideas?
- What seems most promising for the goal?

Output only:
Thought: [Your analysis]
```

**Step > 0 (Refinement):**
```text
You are refining your synthesis.

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
```

**Final Synthesis (Is Final = True):**
```text
You are providing FINAL SYNTHESIS and DECISIONS.

Agenda: {agenda}

Discussion so far:
{context}

Your reasoning process:
{reasoning_steps}

Now write your FINAL SYNTHESIS based on your reasoning.

IMPORTANT:
- Make FINAL DECISIONS, do NOT ask clarifying questions
- Synthesize what the team proposed into a clear action plan
- Be decisive and specific about what to implement
- Structure it clearly for the coding agent to understand

Keep it focused (1-2 paragraphs).
```

### `TeamMeeting` Metadata Extraction
```text
Based on this meeting transcript, extract structured metadata:

{context}

Provide in JSON format:
{
    "key_insights": ["insight 1", "insight 2", ...],
    "decisions": ["decision 1", "decision 2", ...],
    "action_items": ["action 1", "action 2", ...]
}
```

### `IndividualMeeting._react_coding_task`
**Step 0 (Approach):**
```text
You are planning how to implement a coding task.

Task: {task}

Think step-by-step about the APPROACH:
- What's the overall architecture/structure?
- What are the main steps?
- What libraries/methods will you use?

Output only:
Thought: [Your thinking about the overall approach]
```

**Step 1 (Implementation Details):**
```text
You are refining your implementation plan.

Task: {task}
{previous_thoughts}

Think step-by-step about IMPLEMENTATION DETAILS:
- What edge cases need handling?
- What's the data flow?
- What features/transformations are needed?

Output only:
Thought: [Your thinking about implementation details]
```

**Step 2 (Final Details):**
```text
You are finalizing your implementation plan.

Task: {task}
{previous_thoughts}

Think step-by-step about FINAL DETAILS:
- Are there any missing pieces?
- How will you ensure correctness?
- Any optimizations needed?

Output only:
Thought: [Your final thoughts before coding]
```

**Final Code Generation:**
```text
You are implementing a coding task.

Task: {task}

Your reasoning process:
{...}

Now write the COMPLETE, EXECUTABLE code based on your reasoning.

Important constraints:
- Do NOT create or overwrite any preloaded data variables; assume they already exist.
- Do NOT construct dummy toy datasets for these variables.

Output ONLY the code in ```python blocks.
```

### `IndividualMeeting._optional_react_proposal` & `_optional_search_task`
**Initial Prompt:**
```text
You are {agent.title} participating in a team meeting. [Or "Task: {task}"]

Agenda: {agenda} [Or Task details]

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
```

**Follow-up Prompt (after search):**
```text
You searched for papers and got results:

{observation}

Your reasoning so far:
{react_history}

Now provide your final proposal based on your expertise and the papers you found.
Cite papers as supporting evidence: (Author et al., Year)

Keep it focused (1-2 paragraphs).
```

### `IndividualMeeting._search_and_observe` (Paper Analysis)
```text
Extract 2-3 key actionable insights from this paper abstract.

Title: {paper.title}
Abstract: {paper.abstract}

Output ONLY a JSON array of 2-3 brief insights:
["insight 1", "insight 2", "insight 3"]

Focus on methods, findings, or techniques that could be applied.
```

## 3. Orchestrator Prompts (`src/dream_team/orchestrator.py`)

### Bootstrap Exploration
**Exploration Task:**
```text
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
```

**Code Task:**
```text
The PI wants to do initial exploration. Write Python code to implement this:

## PI's Request:
{exploration_plan}

## Problem Statement (for reference):
{problem_statement}

## Available in execution context:
- Pre-imported libraries: pandas (pd), numpy (np), torch, pathlib.Path
- Variables: {list(self.executor.data_context.keys())}
  (You can use any of these variables directly in your code)

## Requirements:
- Inspect dataframes: print(df.info()), df.head(), df.describe(), df.columns
- ONLY print what you observe - no summaries, interpretations, or conclusions
- Use variables from "Available in execution context" above
- Suppress warnings if needed

Output ONLY the Python code, wrapped in ```python code blocks.
```

### Recruitment
**Recruitment Task:**
```text
Based on the problem and exploration results, decide what expertise you need on your team.

## Problem:
{problem_statement}

## Exploration Insights:
**Summary:**
{exploration_analysis.raw_summary}

**Key Observations:**
{exploration_analysis.key_observations}

## Your Task:
List 1-3 team members you want to recruit. For each, provide:
- Title (e.g., "ML Strategist", "Domain Expert", "Data Analyst")
- Expertise (what they should know)
- Role (what they'll contribute)

Be specific about the skills needed based on what you learned from exploration and research papers.

Format your response as a simple list, one team member per line.
```

**Parsing Task:**
```text
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
    "title": "Agent Title",
    "expertise": "Expertise description",
    "role": "Role description"
  }
]
```

### Team Planning Meeting Agenda
```text
**BE CONCISE.**

## Problem:
{problem_statement}

## Objective:
Decide on the next step to improve {self.target_metric}.

## Constraints:
- Be specific and actionable.
- You don't have to solve the entire problem in one iteration. Go step by step, adapt as new information comes in.

## Available Dataframes:
{list(self.executor.data_context.keys())}
{columns_summary}

{base_context}

## Roles:
- **Team Members**: Review recent iterations and propose what to do next.
- **Lead**: Synthesize into a decisive action plan.
```

### Implementation Task
```text
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
```

## 4. Analyzer Prompts (`src/dream_team/analyzers.py`)

### `OutputAnalyzer.analyze`
```text
Analyze the results of a code execution task.

## Task Description:
{task_description}

## Code Executed:
{code}

## Execution Output:
{output}

## Execution Error (if any):
{error}
{traceback}

## Your Goal:
Provide a structured analysis of what happened.
1. Did it succeed?
2. What are the key findings/observations?
3. Were there any errors or warnings?

Output ONLY JSON:
{
    "summary": "Brief summary of what happened",
    "key_observations": ["obs1", "obs2", ...],
    "errors": ["error1", ...],
    "warnings": ["warning1", ...]
}
```

## 5. Evolution Agent Prompts (`src/dream_team/evolution_agent.py`)

### `_build_domain_concepts_from_llm`
```text
Identify 5-7 core technical concepts for this research problem.

## Problem:
{problem_statement}

## Target Metric:
{target_metric}
{context_section}
## Task:
Identify the 5-7 MOST CRITICAL technical concepts for this specific problem.
Focus on strategic concepts that represent distinct areas of expertise.

Output format (one per line):
concept_name | importance (1-3)

Examples: ensemble_methods, time_series_analysis, feature_engineering,
          class_imbalance, regularization, interpretability

Be selective - quality over quantity.
```
