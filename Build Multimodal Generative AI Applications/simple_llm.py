from langchain_ollama import ChatOllama
from langchain_core.prompt import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def remove_nonascii(s):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    system_prompt = """
        You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
     financial products and common financial terms are in the correct format. For each
     financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
     transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to  and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed
    """

    prompt_input = system_prompt + "\n" + ascii_transcript

    messages = [
        {
            "role":"user",
            "content":prompt_input
        }
    ]

    llama32 = ChatOllama(
        model="llama3.2:3b",
        temperature=0.2,
        top_p=0.6
    )

    response = llama32.invoke(messages)
    print(response)

template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOllama(
    mode="minicpm-v4.6:1b",
    temeprature=0.7,
)
chain = (
    {"context": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser
)