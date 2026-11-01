"""The look of the app: one stylesheet and the brand header. Everything here is a FIXED string written by us: nothing a user, a model or a news feed supplied is ever
placed into HTML or CSS, because unsafe_allow_html turns that text into live page code.

Selectors use Streamlit's `data-testid` attributes, which are stable across versions; if one ever stops matching the page simply looks a bit plainer, nothing breaks.
"""
import streamlit as st

LOGO = ('<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#14B8A6"/><stop offset="1" stop-color="#0F766E"/></linearGradient></defs><rect width="34" height="34" rx="9" fill="url(#g)"/>'
        '<polyline points="7,23 13,16 18,20 27,10" fill="none" stroke="white" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="27" cy="10" r="2.4" fill="white"/></svg>')

CSS = """
<style>
  :root { --brand:#0F766E; --brand-dark:#115E59; --ink:#0F172A; --muted:#64748B; --line:#D7DEE7; --card:#FFFFFF; --soft:#EAF4F3; }

  /* page frame: a comfortable reading width and room at the bottom */
  .block-container { max-width: 1180px; padding-top: 1.1rem; padding-bottom: 4rem; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] { display: none; }
  [data-testid="stSidebarCollapsedControl"] { display: none; }

  /* headings: tighter and smaller than Streamlit's very large defaults */
  h1 { font-size: 2rem !important; letter-spacing: -0.02em; }
  h2, [data-testid="stHeading"] h2 { font-size: 1.45rem !important; letter-spacing: -0.01em; margin-top: .4rem; }
  h3 { font-size: 1.12rem !important; }

  /* brand header */
  .brand { display:flex; align-items:center; gap:.7rem; padding:.35rem 0 .1rem 0; }
  .brand .name { font-size:1.45rem; font-weight:750; letter-spacing:-0.02em; color:var(--ink); line-height:1.1; }
  .brand .tag { font-size:.82rem; color:var(--muted); line-height:1.1; margin-top:.15rem; }
  .pill { display:inline-flex; align-items:center; gap:.4rem; font-size:.8rem; font-weight:600; padding:.22rem .65rem; border-radius:999px; border:1px solid var(--line); background:var(--card); color:var(--muted); }
  .pill .dot { width:.55rem; height:.55rem; border-radius:50%; background:#94A3B8; }
  .pill.ok .dot { background:#16A34A; } .pill.bad .dot { background:#DC2626; }
  .disclaimer { font-size:.8rem; color:var(--muted); padding:.2rem 0 .6rem 0; }

  /* tabs */
  [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--line); }
  button[data-baseweb="tab"] { font-weight:600; padding:.65rem 1rem; color:var(--muted); }
  button[data-baseweb="tab"][aria-selected="true"] { color:var(--brand); }
  [data-baseweb="tab-highlight"] { background-color: var(--brand) !important; height:3px; border-radius:3px 3px 0 0; }

  /* cards: metrics, bordered containers (forms, news items), expanders */
  [data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:.8rem; padding:.9rem 1.1rem; box-shadow:0 1px 2px rgba(15,23,42,.05); }
  [data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
  [data-testid="stMetricValue"] { font-size:1.65rem; font-weight:700; letter-spacing:-0.01em; }
  [data-testid="stVerticalBlockBorderWrapper"] { background:var(--card); border-radius:.8rem; box-shadow:0 1px 2px rgba(15,23,42,.04); }
  [data-testid="stExpander"] { background:var(--card); border-radius:.8rem; border-color:var(--line); box-shadow:0 1px 2px rgba(15,23,42,.04); }
  [data-testid="stExpander"] summary { font-weight:600; }
  [data-testid="stForm"] { background:var(--card); border-radius:.8rem; border-color:var(--line); }

  /* buttons: rounded, a touch of lift on hover; starter questions look like chips */
  .stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stPopover"] > button { border-radius:.6rem; font-weight:600; transition: box-shadow .15s, transform .15s; }
  .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { box-shadow:0 2px 8px rgba(15,23,42,.12); }
  .starters .stButton > button, [class*="st-key-starter"] button { border-radius:999px; background:var(--soft); border-color:#BFE0DC; color:var(--brand-dark); font-weight:600; min-height:2.6rem; }
  [class*="st-key-starter"] button:hover { background:#DCEFEC; border-color:var(--brand); }

  /* chat */
  [data-testid="stChatInput"] { border-radius:.9rem; box-shadow:0 2px 10px rgba(15,23,42,.08); }
  [data-testid="stChatMessage"] { border-radius:.9rem; }
  .welcome h2 { margin-bottom:.1rem; } .welcome p { color:var(--muted); margin-top:0; }

  /* login */
  .hero h1 { font-size:2.3rem !important; line-height:1.15; margin:.6rem 0 .5rem 0; }
  .hero p.lead { font-size:1.08rem; color:#334155; max-width:30rem; }
  .hero ul { list-style:none; padding:0; margin:1.2rem 0 0 0; }
  .hero li { display:flex; gap:.7rem; align-items:flex-start; margin:.85rem 0; color:#334155; }
  .hero li .ic { flex:0 0 2rem; height:2rem; border-radius:.6rem; background:var(--soft); color:var(--brand); display:flex; align-items:center; justify-content:center; font-weight:700; }
  .hero li b { color:var(--ink); }

  /* inline code (ids): quiet, in the brand colour rather than Streamlit's bright green */
  code { color:var(--brand-dark) !important; background:var(--soft) !important; }

  /* links: calm until hovered */
  a { text-decoration: none !important; font-weight:600; } a:hover { text-decoration: underline !important; }

  /* data tables */
  [data-testid="stDataFrame"] { border-radius:.7rem; overflow:hidden; }

  /* small screens: tighter page, tabs scroll instead of wrapping */
  @media (max-width: 640px) {
    .block-container { padding-left:.9rem; padding-right:.9rem; }
    h1 { font-size:1.6rem !important; } .hero h1 { font-size:1.75rem !important; }
    button[data-baseweb="tab"] { padding:.55rem .7rem; }
  }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand_header() -> None:
    st.markdown(f'<div class="brand">{LOGO}<div><div class="name">Finnie</div><div class="tag">Your plain-English finance tutor</div></div></div>', unsafe_allow_html=True)


def status_pill(connected: bool) -> None:
    st.markdown(f'<span class="pill {"ok" if connected else "bad"}"><span class="dot"></span>{"Connected" if connected else "Server unavailable"}</span>', unsafe_allow_html=True)


def disclaimer() -> None:
    """Always the same fixed sentence from ui_common (nothing is passed in): it goes into HTML, so it must never carry user text."""
    from src.web_app.ui_common import DISCLAIMER
    st.markdown(f'<div class="disclaimer">ⓘ {DISCLAIMER}</div>', unsafe_allow_html=True)
