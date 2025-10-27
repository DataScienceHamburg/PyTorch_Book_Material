#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

#%% pydantic model
class MyMovieOutput(BaseModel):
    title: str
    main_character: str
    director: str
    release_year: str


# %% prompt
parser = PydanticOutputParser(pydantic_object=MyMovieOutput)
messages = [
    ("system", "Du bist ein Filmexperte. {format_instructions}"),
    ("user", "Handlung: {plot}")
]
prompt_template = ChatPromptTemplate.from_messages(messages).partial(
    format_instructions=parser.get_format_instructions()
)
#%% model 
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
model = ChatGroq(model=MODEL_NAME, temperature=0.2)
#%% chain
chain = prompt_template | model | parser
# %%
chain_inputs = {"plot": "mars, botanik"}
res = chain.invoke(chain_inputs)

#%%
res.model_dump()
# %%
