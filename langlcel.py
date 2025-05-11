from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.globals import set_debug
from langchain.chains.sequential import SimpleSequentialChain
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
from dotenv import load_dotenv,find_dotenv

# Carregando .env
load_dotenv()
#set_debug(True)


class Destino(BaseModel):
    cidade: str = Field(description="Cidade a visitar")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar a cidade")
    latitude: float =  Field(description="coordenadas de latitude")
    longitude: float =  Field(description="coordenadas de longitude")



llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY")
                 ,model="gpt-3.5-turbo"
                 ,temperature=0.7
                ,verbose=True
                 )

parseador = JsonOutputParser(pydantic_object=Destino)

model_city = ChatPromptTemplate.from_template(
    template="""Sugira uma cidade brasileira dado meu interesse por {interesse}
    {formatacao_de_saida}
    """
).partial(formatacao_de_saida=parseador.get_format_instructions())

model_restaurante = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares e famosos em {cidade}"
)

culture_model = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais baseados na {cidade}"
)

final_model = ChatPromptTemplate.from_messages(
    [("ai", "sugestão de viagem para a cidade:{cidade}"),
    ("ai","Restaurantes na cidade: {restaurantes}"),
    ("ai""Atividades culturais presentes na cidade {locais_culturais}"),
    ("system", "combine as informações anteriores em 2 paragráfos coerentes")]
)

get_cidade = RunnableLambda(lambda d: d["cidade"])

part1 = model_city | llm | parseador
part2 = model_restaurante | llm | StrOutputParser()
part3 = culture_model | llm | StrOutputParser()
part4 = final_model | llm | StrOutputParser()

chain = part1 | {"restaurantes":part2,
                 "locais_culturais": part3, "cidade":get_cidade} | part4

request = chain.invoke(input={"interesse":"metropole"})
print(request)


