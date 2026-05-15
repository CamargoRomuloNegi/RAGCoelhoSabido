import streamlit as st
from supabase import create_client
from google import genai

# 1. CONFIGURAÇÃO DA PÁGINA E PREVENÇÃO DE ERROS VISUAIS
st.set_page_config(page_title="Especialista Tributário", page_icon="⚖️")
st.title("⚖️ Assistente RAG: Reforma Tributária (IBS/CBS)")
st.markdown("Faça perguntas sobre o Regulamento do IBS (Resolução CGIBS Nº 6).")

# 2. BARRA LATERAL PARA INSERIR AS CHAVES (Segurança)
with st.sidebar:
    st.header("🔑 Configurações")
    st.write("Insira suas chaves para conectar o banco e a IA:")
    SUPABASE_URL = st.text_input("URL do Supabase", type="password")
    SUPABASE_KEY = st.text_input("Chave Service Role (Supabase)", type="password")
    GOOGLE_API_KEY = st.text_input("Chave Google API (Gemini)", type="password")

# Interrompe a execução com um aviso limpo se faltar alguma chave
if not (SUPABASE_URL and SUPABASE_KEY and GOOGLE_API_KEY):
    st.warning("👈 Por favor, preencha as chaves na barra lateral para liberar o Chat.")
    st.stop()

# 3. INICIALIZANDO AS CONEXÕES COM CACHE PARA PREVENIR LENTIDÃO
@st.cache_resource
def init_clients(url, key, ai_key):
    return create_client(url, key), genai.Client(api_key=ai_key)

supabase, ai_client = init_clients(SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY)

# 4. MEMÓRIA DO CHAT (Histórico)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderiza as mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. O FLUXO DO RAG (Quando o usuário faz uma pergunta)
if pergunta := st.chat_input("Ex: Como funciona o cálculo da base de cálculo do IBS?"):
    
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando artigos no regulamento..."):
            try:
                # PASSO A: Transformar a pergunta em matemática 
                # (Usando o modelo gemini-embedding-2 configurado para 3072 dimensões)
                resposta_emb = ai_client.models.embed_content(
                    model="gemini-embedding-2", 
                    contents=pergunta,
                    config={"output_dimensionality": 3072}
                )
                
                # CORREÇÃO ESTRUTURAL APLICADA: Acessando o índice  da lista de embeddings
                vetor_pergunta = resposta_emb.embeddings[0].values

                # PASSO B: Buscar no Supabase os artigos mais próximos
                busca_supabase = supabase.rpc(
                    'buscar_artigos_ibs',
                    {
                        'query_embedding': vetor_pergunta,
                        'match_threshold': 0.3, 
                        'match_count': 4        
                    }
                ).execute()

                artigos = busca_supabase.data

                # PASSO C: Tratamento de exceção caso o Supabase não retorne dados
                if not artigos:
                    resposta = "Não encontrei artigos no regulamento que respondam a isso."
                    st.markdown(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                else:
                    # PASSO D: Montar o Contexto (Juntar os artigos encontrados)
                    contexto_juridico = ""
                    referencias = []
                    for art in artigos:
                        titulo = art['metadata'].get('titulo', '')
                        capitulo = art['metadata'].get('capitulo', '')
                        contexto_juridico += f"\n[{titulo} - {capitulo}]\n{art['page_content']}\n"
                        referencias.append(art['id'].replace('_', ' '))

                    # PASSO E: Pedir para o Gemini responder baseado EXCLUSIVAMENTE nos artigos
                    prompt_final = f"""
                    Você é um contador e advogado especialista na Reforma Tributária (IBS/CBS).
                    Responda à pergunta do usuário de forma clara, pedagógica e técnica, baseando-se EXCLUSIVAMENTE nos trechos da resolução fornecidos abaixo.
                    Sempre cite de qual Artigo você tirou a informação.
                    Se a resposta não estiver nos trechos fornecidos, diga que a lei não especifica isso.

                    PERGUNTA DO USUÁRIO: {pergunta}

                    TRECHOS DO REGULAMENTO:
                    {contexto_juridico}
                    """

                    resposta_llm = ai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_final
                    )
                    
                    # PASSO F: Exibe a resposta final na tela e salva no histórico
                    resposta_completa = f"{resposta_llm.text}\n\n*(Artigos lidos para esta resposta: {', '.join(referencias)})*"
                    st.markdown(resposta_completa)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_completa})

            except Exception as e:
                # Retorna o erro exato na tela em caso de falha de conexão para evitar que o app trave
                st.error(f"Erro no processamento: {str(e)}")
