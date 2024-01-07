from agents.tools.procedure.dispatcher import ProcedureAgent


class Agent:

    def __init__(self, llm, memory, chat_history, verbose):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose
        
    def run(self, input):
        procedure_agent = ProcedureAgent(llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        return procedure_agent.run(input)
# def run (input, llm, memory, chat_history, verbose):
#     procedure_agent = ProcedureAgent(llm=llm, memory=memory, chat_history=chat_history, verbose=verbose)
#     return procedure_agent.run(input)

