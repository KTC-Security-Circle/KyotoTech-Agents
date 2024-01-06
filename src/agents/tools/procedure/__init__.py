from procedure.dispatcher import agent



def run (input, llm, memory, chat_history, verbose):
    return agent.run(input, llm, memory, chat_history, verbose)

