"""Tab 5: Knowledge. Streamlit concepts: a radio OUTSIDE forms to switch modes, st.cache_data for lists that rarely change, st.selectbox, st.markdown for articles."""
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import guarded, show_error

ALL = "All topics"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_articles(_client: FinnieClient, url: str) -> list:
    return _client.articles()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_article(_client: FinnieClient, url: str, doc_id: str) -> dict:
    return _client.article(doc_id)


@st.cache_data(ttl=300, show_spinner=False)
def cached_search(_client: FinnieClient, url: str, query: str, category: str | None) -> dict:
    return _client.search(query, category)


def pretty(category: str) -> str:
    return category.replace("_", " ").capitalize()


def search_section(client: FinnieClient, categories: list[str]) -> None:
    with st.form("knowledge_search"):
        query = st.text_input("What do you want to learn about?", placeholder="e.g. what is an expense ratio", max_chars=300)
        topic = st.selectbox("Topic", [ALL] + [pretty(c) for c in categories], key="search_topic")
        submitted = st.form_submit_button("Search")
    if submitted:
        st.session_state.knowledge_query = (query.strip(), None if topic == ALL else categories[[pretty(c) for c in categories].index(topic)])
        st.session_state.knowledge_forced = True
    request = st.session_state.get("knowledge_query")
    if not request:
        return
    query, category = request
    if len(query) < 2:
        st.warning("Please type at least two characters.")
        return
    try:
        with st.spinner("Searching…"):
            found = guarded(("search", query, category), lambda: cached_search(client, client.base_url, query, category),
                            force=st.session_state.pop("knowledge_forced", False))
    except ApiError as error:
        show_error(error)
        return
    if not found["answerable"]:
        st.info("Nothing in the knowledge base is a close match for that. Try different words, or ask in the Chat tab.")
        return
    for passage in found["results"]:
        with st.container(border=True):
            st.markdown(f"**{passage['title']}** · {passage['section']}")
            st.markdown(passage["text"])
            st.caption(f"Source: [{passage['source_name']}]({passage['source_url']}) · updated {passage['last_updated']} · match score {passage['similarity']:.2f}")


def browse_section(client: FinnieClient, articles: list[dict], categories: list[str]) -> None:
    topic = st.selectbox("Topic", [ALL] + [pretty(c) for c in categories], key="browse_topic")
    shown = [a for a in articles if topic == ALL or pretty(a["category"]) == topic]
    if not shown:
        st.info("No articles in that topic.")
        return
    labels = {f"{a['title']}  ({pretty(a['category'])}, {a['difficulty'] or 'n/a'})": a["id"] for a in shown}
    choice = st.selectbox("Article", list(labels), key="browse_article")
    try:
        article = guarded(("article", labels[choice]), lambda: cached_article(client, client.base_url, labels[choice]))
    except ApiError as error:
        show_error(error)
        return
    st.markdown(article["text"])
    source = f"[{article['source_name']}]({article['source_url']})" if article.get("source_url") else (article.get("source_name") or "not recorded")
    st.caption(f"Source: {source} · last updated {article.get('last_updated') or 'unknown'} · This article is a plain-language summary; check the source for the full, current text.")


def render(client: FinnieClient) -> None:
    try:
        articles = guarded("articles", lambda: cached_articles(client, client.base_url))
    except ApiError as error:
        show_error(error)
        return
    categories = sorted({a["category"] for a in articles if a["category"]})
    st.caption(f"{len(articles)} short articles in {len(categories)} topics, each with its source.")
    mode = st.radio("How do you want to explore?", ["Search", "Browse articles"], horizontal=True, label_visibility="collapsed")
    if mode == "Search":
        search_section(client, categories)
    else:
        browse_section(client, articles, categories)
