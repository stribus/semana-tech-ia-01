import json 
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv

load_dotenv()

#import streamlit as st


# IMPORTANDO OPENAI LLM - GPT
os.environ['OPENAI_API_KEY']=os.getenv('OPEN_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo")


# CRIANDO YAHOO FINANCE TOOL 
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Obtém os preços das ações para {ticket} do último ano sobre uma empresa específica da API do Yahoo Finance",
    func= lambda ticket: fetch_stock_price(ticket)
)


# CRIANDO DUCKDUCKGO SEARCH TOOL
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


stockPriceAnalyst = Agent(
    role="Analista de Ações",
    goal="Encontrar o preço da ação {ticket} e analisar as tendências.",
    backstory="""Você é altamente experiente em analisar o preço de uma ação específica e fazer previsões sobre seu preço futuro.""",
    verbose=True,
    llm= llm,
    max_iter= 5,
    memory= True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)



getStockPrice = Task(
    description= "Analisar o histórico de preços da ação {ticket} e criar uma análise de tendência de alta, baixa ou lateral",
    expected_output = """" Especifique a tendência atual do preço da ação - alta, baixa ou lateral.
    ex. ação= 'APPL, preço ALTA'
""",
    agent= stockPriceAnalyst
)



newsAnalyst = Agent(
    role="Analista de Notícias de Ações",
    goal="""Criar um resumo curto das notícias de mercado relacionadas à empresa de ações {ticket}. Especifique a tendência atual - alta, baixa ou lateral com
    o contexto das notícias. Para cada ativo de ações solicitado, especifique um número entre 0 e 100, onde 0 é medo extremo e 100 é ganância extrema.""",
    backstory="""Você é altamente experiente em analisar as tendências e notícias do mercado e acompanha ativos há mais de 10 anos.

    Você também é um analista de nível mestre nos mercados tradicionais e tem um profundo entendimento da psicologia humana.

    Você entende notícias, seus títulos e informações, mas olha para elas com uma boa dose de ceticismo.
    Você também considera a fonte dos artigos de notícias.
    """,
    verbose=True,
    llm= llm,
    max_iter= 10,
    memory= True,
    tools=[search_tool],
    allow_delegation=False
)


get_news = Task(
    description= f"""Pegue a ação e pesquise as notícias de mercado relacionadas a ela.
    Assim como noticias relacionadas ao S&P500 e ao mercado geral.
    Use a ferramenta de pesquisa para pesquisar cada uma individualmente.

    A data atual é {datetime.now()}.

    Componha os resultados em um relatório útil""",
    expected_output = """"Um resumo do mercado geral e um resumo de uma frase para cada ativo solicitado.
    Inclua uma pontuação de medo/ganância para cada ativo com base nas notícias. Use o formato:
    <ATIVO DE AÇÃO>
    <RESUMO COM BASE NAS NOTÍCIAS>
    <PREVISÃO DE TENDÊNCIA>
    <PONTUAÇÃO DE MEDO/GANÂNCIA>
""",
    agent= newsAnalyst
)




stockAnalystWrite = Agent(
    role = "Analista de Ações senior",
    goal= """Use a tendência de preço da ação e o relatório de notícias da ação para criar uma análise e escrever o boletim informativo sobre a empresa {ticket} que seja breve e destaque os pontos mais importantes. Foque na tendência de preço da ação, notícias e pontuação de medo/ganância. Quais são as considerações para o futuro próximo? Inclua as análises anteriores da tendência de ações e resumo de notícias.""",
    backstory= """Você é amplamente aceito como o melhor analista de ações do mercado. Você entende conceitos complexos e cria histórias e narrativas convincentes que ressoam com um público mais amplo. Você entende fatores macro e combina várias teorias - por exemplo, teoria do ciclo e análises fundamentais. Você é capaz de manter várias opiniões ao analisar qualquer coisa.""",
    verbose = True,
    llm=llm,
    max_iter = 5,
    memory=True,
    allow_delegation = True
)

writeAnalyses = Task(
    description = """Use a tendência de preço da ação e o relatório de notícias da ação para criar uma análise e escrever o boletim informativo sobre a empresa {ticket}
    que seja breve e destaque os pontos mais importantes.
    Foque na tendência de preço da ação, notícias e pontuação de medo/ganância. Quais são as considerações para o futuro próximo?
    Inclua as análises anteriores da tendência de ações e resumo de notícias.
""",
    expected_output= """"Um boletim informativo de 3 parágrafos eloquente formatado como markdown de maneira fácil de ler. Deve conter:

    - 3 resumos executivos em tópicos
    - Introdução - defina a imagem geral e aumente o interesse
    - a parte principal fornece a essência da análise, incluindo o resumo de notícias e pontuações de medo/ganância
    - resumo - fatos-chave e previsão concreta de tendência futura - alta, baixa ou lateral.
    respostas em português.
""",
    agent = stockAnalystWrite,
    context = [getStockPrice, get_news]
)




crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, get_news, writeAnalyses],
    verbose = 2,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

results= crew.kickoff(inputs={'ticket': 'ITUB3.SA'})

print(results)