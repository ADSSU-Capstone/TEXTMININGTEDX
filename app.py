import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import io
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from scipy.stats import kruskal
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.set_page_config(layout="wide", page_title="TEDx Engagement Analysis",
                   page_icon="🎙️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

:root{
    --bg-0:#12171d;
    --bg-1:#171d24;
    --surface:#222a33;
    --surface-2:#2b3540;
    --border:#38434f;
    --text:#e2e8ee;
    --muted:#93a4b3;
    --ember:#d2691e;
    --ember-2:#f5883a;
    --brass:#c9a227;
    --steel:#5b7f9e;
    --steel-2:#8ab4d6;
}

html, body, [class*="css"]{ font-family:'Barlow', sans-serif; }

.stApp{
    background:
        radial-gradient(1000px 520px at 10% -8%, rgba(210,105,30,.16), transparent 62%),
        radial-gradient(900px 480px at 95% 0%, rgba(91,127,158,.20), transparent 60%),
        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
    color: var(--text);
}

header[data-testid="stHeader"]{
    background: rgba(18,23,29,.55);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(56,67,79,.55);
}

.block-container{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }

h1,h2,h3,h4,h5,h6{
    font-family:'Barlow Condensed', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    color: var(--text);
    margin-bottom: .4rem;
}
h2{
    font-size: 1.85rem;
    border-left: 5px solid var(--ember);
    padding-left: 14px;
    color: #fff;
}
h3{
    font-size: 1.25rem;
    color: var(--steel-2);
    border-bottom: 1px solid rgba(56,67,79,.9);
    padding-bottom: 6px;
}
p, li, label, span{ color: var(--text); }

.hero{
    position: relative;
    padding: 28px 34px 26px;
    border-radius: 16px;
    border: 1px solid var(--border);
    background:
        linear-gradient(120deg, rgba(210,105,30,.22) 0%, rgba(91,127,158,.16) 48%, rgba(34,42,51,.85) 100%);
    box-shadow: 0 18px 44px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.05);
    overflow: hidden;
    margin-bottom: 6px;
}
.hero::after{
    content:"";
    position:absolute; inset:0;
    background: repeating-linear-gradient(135deg, rgba(255,255,255,.035) 0 2px, transparent 2px 14px);
    pointer-events:none;
}
.hero-tag{
    display:inline-block;
    font-family:'Barlow Condensed', sans-serif;
    font-size:.82rem; letter-spacing:3px; font-weight:700;
    color:#14181d;
    background: linear-gradient(180deg, var(--ember-2), var(--ember));
    padding: 4px 12px; border-radius: 4px;
    margin-bottom: 12px;
}
.hero-title{
    font-family:'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    line-height: 1;
    letter-spacing: 3px;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #ffffff 0%, var(--ember-2) 55%, var(--brass) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub{
    color: var(--muted);
    font-size: 1.02rem;
    letter-spacing: .6px;
    margin: 0;
}
.hero-bar{
    margin-top: 18px;
    height: 4px; width: 130px; border-radius: 3px;
    background: linear-gradient(90deg, var(--ember), var(--brass), var(--steel));
    animation: pulseBar 2.6s ease-in-out infinite;
}
@keyframes pulseBar{
    0%,100%{ width:130px; opacity:1; }
    50%    { width:240px; opacity:.65; }
}

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #1b222a 0%, #131920 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{ color: var(--ember-2); border:none; padding-left:0; }

.side-head{
    font-family:'Barlow Condensed', sans-serif;
    font-size: .85rem;
    letter-spacing: 3px;
    font-weight: 700;
    color: var(--steel-2);
    padding: 14px 0 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
}

section[data-testid="stFileUploaderDropzone"]{
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px;
    transition: border-color .2s ease, box-shadow .2s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover{
    border-color: var(--ember) !important;
    box-shadow: 0 0 0 3px rgba(210,105,30,.12);
}
div[data-baseweb="select"] > div{
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    transition: border-color .2s ease;
}
div[data-baseweb="select"] > div:hover{ border-color: var(--ember) !important; }
span[data-baseweb="tag"]{
    background: linear-gradient(180deg, var(--ember-2), var(--ember)) !important;
    color: #14181d !important;
    font-weight: 600;
}

div[data-testid="stTabs"] div[data-baseweb="tab-list"]{
    gap: 4px;
    border-bottom: 2px solid var(--border);
}
button[data-baseweb="tab"]{
    font-family:'Barlow Condensed', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 600;
    font-size: 1rem;
    color: var(--muted);
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    transition: all .18s ease;
}
button[data-baseweb="tab"]:hover{
    color: var(--text);
    background: rgba(210,105,30,.10);
    border-color: rgba(210,105,30,.28);
}
button[data-baseweb="tab"][aria-selected="true"]{
    color: #14181d;
    background: linear-gradient(180deg, var(--ember-2), var(--ember));
    border-color: var(--ember);
    box-shadow: 0 -2px 20px rgba(210,105,30,.40);
}
div[data-baseweb="tab-highlight"]{ display:none; }
div[data-baseweb="tab-border"]{ background: var(--border); }

div[data-testid="stTabs"] div[role="tabpanel"]{ animation: fadeUp .35s ease both; }
@keyframes fadeUp{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;} }

div[data-testid="stMetric"]{
    position: relative;
    background: linear-gradient(160deg, var(--surface-2) 0%, var(--surface) 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px 16px;
    overflow: hidden;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div[data-testid="stMetric"]::before{
    content:"";
    position:absolute; left:0; top:0; height:3px; width:38%;
    background: linear-gradient(90deg, var(--ember), var(--brass));
    transition: width .32s ease;
}
div[data-testid="stMetric"]:hover{
    transform: translateY(-4px);
    border-color: rgba(210,105,30,.55);
    box-shadow: 0 14px 30px rgba(0,0,0,.50);
}
div[data-testid="stMetric"]:hover::before{ width:100%; }
div[data-testid="stMetricLabel"] p{
    font-family:'Barlow Condensed', sans-serif;
    font-size: .95rem !important;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--muted) !important;
}
div[data-testid="stMetricValue"]{
    font-family:'Bebas Neue', sans-serif;
    font-size: 2.3rem;
    letter-spacing: 1.5px;
    color: #fff;
}

div[data-testid="stDataFrame"]{
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-testid="stDataFrame"]:hover{
    border-color: rgba(91,127,158,.75);
    box-shadow: 0 0 0 1px rgba(91,127,158,.20), 0 12px 26px rgba(0,0,0,.40);
}

div[data-testid="stExpander"]{
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: linear-gradient(160deg, rgba(43,53,64,.75), rgba(34,42,51,.75));
    overflow: hidden;
    transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    margin-bottom: 8px;
}
div[data-testid="stExpander"]:hover{
    border-color: rgba(210,105,30,.60) !important;
    box-shadow: 0 10px 24px rgba(0,0,0,.42);
    transform: translateX(3px);
}
div[data-testid="stExpander"] summary{
    font-family:'Barlow Condensed', sans-serif;
    letter-spacing: 1px;
    font-weight: 600;
    color: var(--text);
    transition: color .2s ease;
}
div[data-testid="stExpander"] summary:hover{ color: var(--ember-2); }

div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"]{
    background: linear-gradient(160deg, rgba(43,53,64,.55), rgba(34,42,51,.55));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-testid="stVegaLiteChart"]:hover{
    border-color: rgba(210,105,30,.45);
    box-shadow: 0 10px 26px rgba(0,0,0,.40);
}

div[data-testid="stAlert"]{
    border-radius: 12px;
    border-left: 4px solid var(--ember);
    background: rgba(210,105,30,.10);
}

::-webkit-scrollbar{ width: 11px; height: 11px; }
::-webkit-scrollbar-track{ background: #10151a; }
::-webkit-scrollbar-thumb{
    background: linear-gradient(180deg, #3d4a57, #2b343f);
    border-radius: 6px;
    border: 2px solid #10151a;
}
::-webkit-scrollbar-thumb:hover{ background: linear-gradient(180deg, var(--ember-2), var(--ember)); }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_example_data():
    df = pd.DataFrame({
        'Speaker': ['Jessica Dere', 'Nadine Burke Harris', 'Guy Winch', 'John Rigg', 'Dr. Jeffrey Lieberman', 'Judson Brewer'],
        'Keyword': ['Challenges and rewards of a culturally informed approach to mental health', 'How childhood trauma affects health across a lifetime', 'How to practice emotional first aid', 'The effect of trauma on the brain and how it affects behaviors', 'Imagine there was no stigma to mental illness', 'A simple way to break a bad habit'],
        'Date Released': pd.to_datetime(['2015-04-21', '2015-02-18', '2015-02-17', '2015-03-21', '2016-01-12', '2016-02-25']),
        'Views': [465557, 8300874, 5467141, 2760468, 609769, 16882137],
        'Likes': ['3K', '197K', '201K', '33K', '9K', '328K'],   # <- matches the real file format
        'Comments': [45, 9237, 3391, 2319, 643, 5667],
        'Topic': ['Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness'],
        'Link': ['https://youtu.be/VrYmQDiunSc', 'https://youtu.be/95ovIJ3dsNk',
                 'https://youtu.be/F2hc2FLOdhI', 'https://youtu.be/m9Pg4K1ZKws',
                 'https://youtu.be/WrbTbB9tTtA', 'https://youtu.be/-moW9jvvMr4'],
        'Transcript': [
            "thank you it's truly an honor to be here I would like to start my talk today with a little exercise introducing you to a person named Pat Pat is 31 years old and lives at home with one sibling and their parents",
            "In the mid-'90s, the CDC and Kaiser Permanente discovered an exposure that dramatically increased the risk for seven out of 10 of the leading causes of death in the United States.",
            "I grew up with my identical twin, who was an incredibly loving brother. Now, one thing about being a twin is, it makes you an expert at spotting favoritism.",
            "I want to ask you to think back to some occasion in your life when you might have gotten in an argument with someone particularly someone that you loved you cared for a lot",
            "Personal Question I know we're only just meeting for the first time, but I'd like to ask you a question, and it's a rather personal question.",
            "When I was first learning to meditate, the instruction was to simply pay attention to my breath, and when my mind wandered, to bring it back."
        ]
    })
    return df


# === NEW: robust parsing helpers ==============================================

def parse_count(val):
    """Turn '197K', '1.6K', '26,935', or a plain int into a real integer.
    Returns NaN only if the value genuinely can't be parsed."""
    if val is None:
        return np.nan
    if isinstance(val, (bool, np.bool_)):
        return np.nan
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return np.nan if np.isnan(val) else int(val)

    s = str(val).strip().replace(",", "").replace(" ", "")
    if s == "" or s.lower() in {"nan", "none", "-", "null"}:
        return np.nan

    mult = 1.0
    suffix = s[-1].upper()
    if suffix == "K":
        mult, s = 1_000.0, s[:-1]
    elif suffix == "M":
        mult, s = 1_000_000.0, s[:-1]
    elif suffix == "B":
        mult, s = 1_000_000_000.0, s[:-1]

    try:
        return int(round(float(s) * mult))
    except ValueError:
        return np.nan


def extract_video_id(link):
    """Get the 11-char YouTube video ID from a URL, ignoring ?si= trackers."""
    if not isinstance(link, str):
        return None
    m = re.search(r"(?:youtu\.be/|watch\?v=)([A-Za-z0-9_-]{11})", link)
    return m.group(1) if m else link

# === END NEW helpers =========================================================


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)


def process_data(df):
    """Clean, filter, dedupe. Preserves all valid talks so the KPI matches
    the row count of the uploaded sheet (minus true duplicates)."""
    df = df.copy()

    # ---- 1. Parse numeric columns properly (handles "197K", "26,935", …) ----
    for col in ['Views', 'Likes', 'Comments']:
        if col in df.columns:
            df[col] = df[col].apply(parse_count)

    # ---- 2. Dates + 2015-2025 scope ----
    df['Date Released'] = pd.to_datetime(df['Date Released'], errors='coerce')
    df = df.dropna(subset=['Date Released'])
    df = df[(df['Date Released'].dt.year >= 2015) &
            (df['Date Released'].dt.year <= 2025)]
    df['year'] = df['Date Released'].dt.year

    bins   = [2014, 2019, 2025]
    labels = ['2015-2019', '2020-2025']
    df['period'] = pd.cut(df['year'], bins=bins, labels=labels,
                          right=True, include_lowest=True)
    df = df.dropna(subset=['period'])

    # ---- 3. Views is required; Likes / Comments default to 0 ----
    df = df.dropna(subset=['Views'])
    df['Likes']    = df['Likes'].fillna(0).astype(int)
    df['Comments'] = df['Comments'].fillna(0).astype(int)
    df['engagement_score'] = df['Views'] + df['Likes'] + df['Comments']

    # ---- 4. Transcripts (need text for the mining tabs) ----
    df = df.dropna(subset=['Transcript'])
    df['clean_transcript'] = df['Transcript'].apply(clean_text)
    df = df[df['clean_transcript'].str.strip() != '']

    # ---- 5. Dedupe on VIDEO ID, not Keyword.
    #         Two different talks can share a title; same video re-uploaded
    #         with a different ?si= param is a true duplicate.
    #         Done AFTER transcript filtering so we keep the row with content.
    if 'Link' in df.columns:
        df['video_id'] = df['Link'].apply(extract_video_id)
        df = df.drop_duplicates(subset=['video_id'], keep='first')

    df = df.reset_index(drop=True)
    return df


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="side-head">DATA SOURCE</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'],
                                         label_visibility="collapsed")
        use_example = st.checkbox("Use example dataset (if no upload)", value=False)

        data = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    # Prefer Sheet1 (matches the provided Book1.xlsx);
                    # fall back to the first sheet if Sheet1 doesn't exist.
                    try:
                        data = pd.read_excel(uploaded_file, sheet_name="Sheet1")
                    except ValueError:
                        data = pd.read_excel(uploaded_file, sheet_name=0)
            except Exception as e:
                st.error(f"Error loading file: {e}")
        elif use_example:
            data = load_example_data()

        if data is None:
            st.info("Awaiting data. Upload a file or tick the example dataset box.")
            st.stop()

        return data


def render_filters(df):
    with st.sidebar:
        st.markdown('<div class="side-head">FILTERS</div>', unsafe_allow_html=True)

        all_periods = list(df['period'].cat.categories)
        selected_periods = st.multiselect(
            "Period", all_periods, default=all_periods,
            help="Restrict every tab to the selected five-year windows."
        )
        if not selected_periods:
            selected_periods = all_periods

        max_eng = int(df['engagement_score'].max()) if len(df) else 1
        step = max(1, max_eng // 100)
        min_eng = st.slider(
            "Minimum engagement score", 0, max_eng, 0, step=step,
            help="Hide low-performing talks from all charts and tables."
        )

    out = df[df['period'].isin(selected_periods)]
    out = out[out['engagement_score'] >= min_eng]

    if out.empty:
        st.warning("No talks match the current filters. Loosen them in the sidebar.")
        st.stop()
    return out


def main():
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">TEDX DATA INTELLIGENCE</div>
        <div class="hero-title">TEDx Talk Engagement Analysis</div>
        <p class="hero-sub">Audience engagement, language patterns and topic clusters · 2015 – 2025</p>
        <div class="hero-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    raw_df = render_sidebar()

    required_cols = ['Keyword', 'Speaker', 'Date Released', 'Views', 'Likes',
                     'Comments', 'Transcript']
    if not all(c in raw_df.columns for c in required_cols):
        st.error(f"Dataset is missing one or more required columns: {required_cols}")
        st.stop()

    df = process_data(raw_df)
    if df.empty:
        st.error("No data available after filtering. Check the 2015-2025 scope, "
                 "English transcripts, and valid engagement metrics.")
        st.stop()

    # === NEW: quiet sanity line — delete once you trust the pipeline =========
    distinct_vid = df['video_id'].nunique() if 'video_id' in df.columns else 'n/a'
    st.caption(
        f"Pipeline check · raw rows: {len(raw_df):,} · "
        f"after cleaning: {len(df):,} · "
        f"distinct videos: {distinct_vid}"
    )
    # === END sanity line =====================================================

    df = render_filters(df)

    peak_period = df.groupby('period', observed=True)['engagement_score'].sum().idxmax()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Talks Analysed", f"{len(df):,}")
    k2.metric("Total Engagement", f"{int(df['engagement_score'].sum()):,}")
    k3.metric("Avg per Talk", f"{int(df['engagement_score'].mean()):,}")
    k4.metric("Peak Period", str(peak_period))
    st.write("")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "① Engagement by Period",
        "② Keyword & Theme Patterns",
        "③ Topic Clusters",
        "④ Trends & Variations",
        "⑤ Top Performers"
    ])

    with tab1:
        st.header("Audience Engagement Analysis")
        st.write("Aggregate audience engagement metrics for each five-year period.")

        period_agg = df.groupby('period', observed=True).agg(
            Videos=('Keyword', 'count'),
            Views=('Views', 'sum'),
            Likes=('Likes', 'sum'),
            Comments=('Comments', 'sum'),
            Avg_Engagement=('engagement_score', 'mean'),
            Total_Engagement=('engagement_score', 'sum')
        ).reset_index()

        st.dataframe(period_agg.style.format({
            'Videos': '{:,}', 'Views': '{:,}', 'Likes': '{:,}',
            'Comments': '{:,}', 'Avg_Engagement': '{:,.0f}',
            'Total_Engagement': '{:,.0f}'
        }), use_container_width=True)

        st.subheader("Total Engagement Score by Period")
        st.bar_chart(period_agg.set_index('period')['Total_Engagement'],
                     use_container_width=True)

        st.subheader("Average Engagement Score per Video")
        st.bar_chart(period_agg.set_index('period')['Avg_Engagement'],
                     use_container_width=True)

        st.subheader("Individual Engagement Metrics")
        st.bar_chart(period_agg.set_index('period')[['Views', 'Likes', 'Comments']],
                     use_container_width=True)

    with tab2:
        st.header("Content Identification via Text Mining")
        st.write("Most frequent terms and recurring bigrams across all selected transcripts.")

        all_text = ' '.join(df['clean_transcript'])
        tokens = all_text.split()

        word_freq = Counter(tokens).most_common(25)
        freq_df = pd.DataFrame(word_freq, columns=['Term', 'Frequency']).set_index('Term')

        st.subheader("Top 25 Most Frequent Keywords")
        st.dataframe(freq_df.T, use_container_width=True)
        st.bar_chart(freq_df, use_container_width=True)

        st.subheader("Top 20 Recurring Bigrams (Themes)")
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        bigram_freq = Counter(bigrams).most_common(20)
        bigram_df = pd.DataFrame(bigram_freq, columns=['Bigram', 'Frequency'])
        bigram_df['Bigram'] = bigram_df['Bigram'].apply(lambda x: f"{x[0]} {x[1]}")
        st.dataframe(bigram_df.set_index('Bigram').T, use_container_width=True)
        st.bar_chart(bigram_df.set_index('Bigram'), use_container_width=True)

    with tab3:
        st.header("Topic Modeling and Engagement")
        st.write("NMF surfaces latent topics; engagement is then compared across clusters.")

        num_topics = min(5, len(df))
        if len(df) < 2:
            st.warning("Insufficient data for topic modeling. At least 2 talks required.")
        else:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, stop_words='english')
            tfidf = tfidf_vectorizer.fit_transform(df['clean_transcript'])

            nmf = NMF(n_components=num_topics, random_state=42,
                      init='nndsvda', max_iter=500)
            nmf_features = nmf.fit_transform(tfidf)
            feature_names = tfidf_vectorizer.get_feature_names_out()

            topic_keywords = {}
            for i, topic in enumerate(nmf.components_):
                top = [feature_names[j] for j in topic.argsort()[:-11:-1]]
                topic_keywords[f"Topic {i+1}"] = ", ".join(top)

            st.subheader("Identified Topics and Keywords")
            st.dataframe(pd.DataFrame.from_dict(
                topic_keywords, orient='index', columns=['Top Keywords']
            ), use_container_width=True)

            df['topic_cluster'] = nmf_features.argmax(axis=1)
            df['topic_label'] = df['topic_cluster'].apply(lambda x: f"Topic {x+1}")

            topic_agg = df.groupby('topic_label').agg(
                Videos=('Keyword', 'count'),
                Total_Engagement=('engagement_score', 'sum'),
                Avg_Engagement=('engagement_score', 'mean')
            ).reset_index().rename(columns={'topic_label': 'Topic'})

            st.subheader("Engagement Metrics by Topic Cluster")
            st.dataframe(topic_agg.style.format({
                'Videos': '{:,}', 'Total_Engagement': '{:,.0f}',
                'Avg_Engagement': '{:,.0f}'
            }), use_container_width=True)

            st.subheader("Average Engagement Score by Topic")
            st.bar_chart(topic_agg.set_index('Topic')['Avg_Engagement'],
                         use_container_width=True)

    with tab4:
        st.header("Trend Analysis Across Periods")
        st.write("Period-over-period movement in aggregate engagement and topic mix.")

        pivot_engagement = df.pivot_table(
            index='period', values=['Views', 'Likes', 'Comments', 'engagement_score'],
            aggfunc='sum', observed=True
        )
        st.subheader("Aggregate Metrics Over Time")
        st.dataframe(pivot_engagement.style.format('{:,.0f}'), use_container_width=True)

        st.subheader("Growth in Engagement Score")
        if len(pivot_engagement) > 1:
            growth = pivot_engagement['engagement_score'].pct_change() * 100
            growth_df = pd.DataFrame({'Period': growth.index, 'Growth (%)': growth.values})
            st.dataframe(growth_df.style.format({'Growth (%)': '{:,.1f}%'}),
                         use_container_width=True)
            st.bar_chart(growth_df.set_index('Period'), use_container_width=True)
        else:
            st.info("At least two periods are required to compute growth.")

        if 'topic_label' in df.columns:
            st.subheader("Topic Share Over Time")
            topic_share = df.groupby(['period', 'topic_label'],
                                     observed=True).size().unstack(fill_value=0)
            topic_share_pct = topic_share.div(topic_share.sum(axis=1), axis=0) * 100
            st.dataframe(topic_share_pct.style.format('{:.1f}%'), use_container_width=True)
            st.bar_chart(topic_share_pct, use_container_width=True)

    with tab5:
        st.header("Top Performing Content")
        st.write("The three highest-engagement talks inside each period.")

        top_videos = (df.sort_values('engagement_score', ascending=False)
                        .groupby('period', observed=True).head(3))

        for period in df['period'].unique():
            st.subheader(f"Period: {period}")
            period_top = top_videos[top_videos['period'] == period]
            for _, row in period_top.iterrows():
                with st.expander(f"{row['Keyword']} — {row['Speaker']}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Views", f"{row['Views']:,}")
                    c2.metric("Likes", f"{row['Likes']:,}")
                    c3.metric("Comments", f"{row['Comments']:,}")
                    c4.metric("Engagement", f"{row['engagement_score']:,}")

        st.subheader("Statistical Comparison of Topic Engagement")
        if 'topic_label' in df.columns and df['topic_label'].nunique() > 1:
            groups = [g['engagement_score'].values
                      for _, g in df.groupby('topic_label')]
            if all(len(g) > 0 for g in groups):
                try:
                    stat, p_value = kruskal(*groups)
                    st.metric("Kruskal-Wallis H-test p-value", f"{p_value:.4f}")
                    if p_value < 0.05:
                        st.write("✅ Engagement differs significantly across topics (p < 0.05).")
                    else:
                        st.write("No statistically significant difference across topics (p ≥ 0.05).")
                except ValueError:
                    st.warning("Unable to run Kruskal-Wallis: each group needs ≥ 1 sample.")
            else:
                st.warning("Insufficient data for the statistical test.")
        else:
            st.info("Topic modeling is required first — widen your filters or load more data.")


if __name__ == "__main__":
    main()
