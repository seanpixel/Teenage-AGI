# Heuristic Orchestration Chain

The chain is meant to operate various agents that have a predetermined set of goals they can change based on their operation, and information gathered about the user and his experiences

## HOC

HOC or heuristic orchestration chain has as a goal to implement a goal given, by choosing one of the various methods available and then optimise the goal itself after the result is produced and assesed

This type of a system requires: 
Based on Newell and Simon (1958) - Report on a general problem-solving program

1. Problem Identification
    Methods: 
    Decomposition
    Analogies
    Root cause analysis
    Goal clarification
    Constraint identification
    SWOT analysis
    Expert consultation
    Visualization
2. Problem Definition
3. Strategy selection
4. Information collection
5. Resource distribution
6. Process monitoring
7. Solution evaluation

`objective` (mandatory) - The overarching objective you want the task orchestration system to converge to
`first_task` (optional) - The prompt it gets for its "first task", which is usually some form of creating a task list. The default is "Make a todo list".

The `from_llm` method that constructs the chain takes in the following arguments that may be of interest:

`llm` - The LLM model you want to the chain to use. Note: Using a model like GPT-4 add up costs extremely quickly. Use with caution.
`vectorstore` - The vectorstore you want the chain to use
`max_iterations` - The maximum number of iterations, i.e. number of tasks that BabyAGI will output a result for and iterate on. If this number is not provided, the chain WILL run forever. 