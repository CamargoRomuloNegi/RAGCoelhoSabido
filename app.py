import streamlit as st
from supabase import create_client
from google import genai

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Especialista Tributário", page_icon="⚖️")
st.title("⚖️ Assistente RAG: Reforma Tributária (IBS/CBS)")
st.markdown("Faça perguntas sobre o Regulamento do IBS (Resolução CGIBS Nº 6).")

# 2. LEITURA DAS CHAVES VIA STREAMLIT SECRETS
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError as e:
    st.error(f"Configuração ausente no Streamlit Secrets: {str(e)}")
    st.stop()

# 3. INICIALIZANDO AS CONEXÕES COM CACHE
@st.cache_resource
def init_clients(url, key, ai_key):
    supabase_client = create_client(url, key)
    ai_client = genai.Client(api_key=ai_key)
    return supabase_client, ai_client

supabase, ai_client = init_clients(SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY)

# 4. MEMÓRIA DO CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. FLUXO DO RAG
if pergunta := st.chat_input("Ex: Como funciona o cálculo da base de cálculo do IBS?"):

    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando artigos no regulamento..."):
            try:
                resposta_emb = ai_client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=pergunta,
                    config={"output_dimensionality": 3072}
                )

                vetor_pergunta = resposta_emb.embeddings[0].values

                busca_supabase = supabase.rpc(
                    "buscar_artigos_ibs",
                    {
                        "query_embedding": vetor_pergunta,
                        "match_threshold": 0.3,
                        "match_count": 4
                    }
                ).execute()

                artigos = busca_supabase.data

                if not artigos:
                    resposta = "Não encontrei artigos no regulamento que respondam a isso."
                    st.markdown(resposta)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": resposta}
                    )
                else:
                    contexto_juridico = ""
                    referencias = []

                    for art in artigos:
                        metadata = art.get("metadata", {}) or {}
                        titulo = metadata.get("titulo", "")
                        capitulo = metadata.get("capitulo", "")

                        contexto_juridico += (
                            f"\n[{titulo} - {capitulo}]\n"
                            f"{art.get('page_content', '')}\n"
                        )

                        referencias.append(str(art.get("id", "")).replace("_", " "))

                    prompt_final = f"""
Você é um contador e advogado especialista na Reforma Tributária (IBS/CBS).

Responda à pergunta do usuário de forma clara, pedagógica e técnica, baseando-se EXCLUSIVAMENTE nos trechos da resolução fornecidos abaixo.

Sempre cite de qual Artigo você tirou a informação.

Se a resposta não estiver nos trechos fornecidos, diga que os trechos recuperados não especificam isso.

PERGUNTA DO USUÁRIO:
{pergunta}

TRECHOS DO REGULAMENTO:
{contexto_juridico}
"""

                    resposta_llm = ai_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt_final
                    )

                    resposta_completa = (
                        f"{resposta_llm.text}\n\n"
                        f"*(Artigos lidos para esta resposta: {', '.join(referencias)})*"
                    )

                    st.markdown(resposta_completa)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": resposta_completa}
                    )

            except Exception as e:
                st.error(f"Erro no processamento: {str(e)}")
