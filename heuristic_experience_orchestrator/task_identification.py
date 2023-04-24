from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM


class TaskIdentificationChain(LLMChain):
    """Chain to generate tasks."""



    @classmethod
    def from_llm(cls, llm: BaseLLM,  verbose: bool = True, value: str = None) -> LLMChain:
        """Get the response parser."""

        def get_template_by_value(self, value):
            if value == "Decomposition":
                template = (
                    """ Hey ChatGPT, I need your help in decomposing the following task into a series of manageable steps for the purpose of task identification based on 
                    Newell and Simon paper. Return the result as a json with the result type 'Identification' and 'Value': 'Decomposition'  : {task_description}"""
                )
            elif value == "Analogy":
                template = (
                    """ Hey ChatGPT, I need your help in creating an analogy for the purpose of task identification based on 
                    Newell and Simon paper. Return the result as a json with the result type 'Identification' and 'Value': 'Analogy'  : {task_description}"""
                )

            elif value == "Template":
                template = (
                    "Template B content"
                )

            elif value == "Templatetest":
                template = (
                    "Template B content"
                )
            else:
                template = (
                    " Return the tasks as an array."
                )
            return template
        if value:
            task_creation_template = get_template_by_value(value)
        else:
            task_creation_template = (
                "Default template content"
            )

        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "task_description"

            ],
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)
