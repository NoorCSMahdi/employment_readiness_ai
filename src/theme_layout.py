import os

def apply_theme(st):
    css_path = os.path.join(os.path.dirname(__file__), "theme.css")
    with open(css_path, "r", encoding="utf-8") as file:
        css = file.read()

    # Keep Streamlit top controls visible by default.
    # Use ?clean=1 only when you want to hide them.
    clean_value = st.query_params.get("clean", "0")
    if isinstance(clean_value, list):
        clean_value = clean_value[0] if clean_value else "0"
    clean_mode = str(clean_value).lower() in {"1", "true", "yes"}
    if not clean_mode:
        css += "\nheader[data-testid='stHeader'], div[data-testid='stToolbar'] { display: block !important; }\n"

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


HEADER_HTML = """
<header class="main-header">
    <div class="header-content">
        <span class="brand-badge">AI</span>
        <div class="title-group">
            <h1>AI CAREER ASSISTANT</h1>
            <p>Career Matching and ATS Review Platform</p>
        </div>
    </div>
</header>
<div class="hero-banner">
    <div class="hero-content">
        <h1>WE BUILD<br/>GREAT CAREERS</h1>
        <p>
            Upload your CV, analyze ATS quality, match relevant jobs,
            and get clear learning steps to close skill gaps.
        </p>
    </div>
</div>
"""


def render_header(st):
    st.markdown(HEADER_HTML, unsafe_allow_html=True)
