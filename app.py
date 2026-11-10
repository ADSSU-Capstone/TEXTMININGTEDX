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
        'Likes': [3000, 197000, 201000, 33000, 9000, 328000],
        'Comments': [45, 9237, 3391, 2319, 643, 5667],
        'Topic': ['Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness', 'Mental Health Awareness'],
        'Transcript': [
            "thank you it's truly an honor to be here I would like to start my talk today with a little exercise introducing you to a person named Pat Pat is 31 years old and lives at home with one sibling and their parents Pat has always lived in the family home and would not consider moving out to live alone except after getting married Pat's parents always play a significant role in making decisions about all aspects of Pat's life including education work decisions Pat's friendships and romantic life this has been the case since Pat was a child Pat rarely makes decisions without consulting family members and rarely expresses disagreement with others out of fear of disapproval or loss of support Pat feels very uncomfortable when alone and tends to cling to others out of fear of being abandoned now take a moment to picture Pat are you picturing a young man a young woman are you picturing a person of a particular ethnic or cultural background maybe similar to your own or different and consider for a moment whether such information would make a difference in how you think about Pat and whether or not you think that Pat might have some kind of mental health concern now what if I added the following information specifying that Pat is Patricia Lee and that she and her family emigrated to Canada from China when she was 5 years old or the following where Pat is Patrick Smith whose maternal and paternal grandparents immigrated to Canada from the United Kingdom and whose parents grew up in the Toronto area judging from some faces that I can see this information indeed makes a difference in how you think about Pat simply adding some basic demographic information can significantly shift how we think about the very same behavior this exercise has elicited similar responses in my classroom when I have used it in teaching about personality disorders which offer a particularly striking example of the need to consider cultural factors when defining mental illness since they are themselves defined by patterns of behavior that deviate from cultural expectations I used it exercise to introduce the role of cultural factors in thinking about mental health and illness and to introduce the central idea that culture matters in my experience once introduced Culture matters to this idea people seem to intuitively know that culture makes a difference when thinking about mental health and mental illness without specifically talking about cultural differences in things like parent child relationships Independence and interdependence or assertiveness we seem to just know that it makes a difference if we're talking about Patricia or Patrick but not only does culture matter when it comes to mental health it is fundamental many people including myself argue that a culturally informed approach is really critical to effective Healthcare and that only by including cultural factors in our discussions of this field will we truly make advances recently a group of medical and social scientists writing in one of the world's most prestigious medical journals stated the following that the systematic neglect of culture in Health and healthc Care is the single biggest barrier to the advancement of the highest standard of Health worldwide a pretty powerful statement so the this is great I see a number of you nodding and you seem to be in agreement that reflecting on Pat illustrates for you at least at a basic level the importance of considering cultural questions when it comes to mental health but stopping here would be problematic for a number of reasons first of all I still have 15 minutes and 52 seconds left on the clock but more importantly stopping here could serve to reinforce certain cultural stereotypes contrasting the lead versus The Smiths and would not really leave you with any idea of how to actually go about considering cultural factors or worse it might leave you with an overly simplistic idea of how to do this suggesting perhaps that simply knowing a person's ethnic background or immigration history provides enough information to really understand the meaning of their behaviors and relationships so acknowledging that culture matters is a crucial initial step sort of like a first piece of a puzzle but we need to go much further so if knowing Pat's cultural Common cultural ideas background is like having a first piece of the puzzle much more needs to be learned before getting even close to a complete picture for example based on common cultural ideas you may have had the sense that the description of Pat's Behavior became somehow more acceptable when it was Patricia a young woman of Chinese Canadian background compared to patri a young man and second generation Canadian of British descent however although interdependence and social harmony are values more commonly endorsed among people of Chinese background compared to Western European background is it necessarily the case that Patricia's parents are happy with her level of dependence upon them maybe their migration to Canada was partially driven by their own traits of Independence and they are really quite concerned that their daughter has not developed greater autonomy or self-reliance maybe Patricia has never felt that she fits in among her peers and really feels incapable of attaining the level of Independence as encouraged by her parents on the other hand although Independence and autonomy are commonly endorsed values in North America maybe it's normative within Patrick's family to defer to one's parents and to view life decisions as communal endeavors Cultural assumptions these examples illustrate that we cannot simply make sweeping assumptions about people purely based on their cultural background nor can we make simple comparisons or distinctions between people simply because they're from different contexts now this might suggest to you then that cultural information is of not much use at all and that we should simply focus on the individual and their unique experiences but this would also be problematic and incomp Complete because to truly understand Patricia Lee's experience we must know that her parents level of emphasis on Independence may be relatively uncommon uh uncommon within their cultural community and on the other hand um or similarly Patrick Smith's experience of his level of dependence upon his family would likely carry different meaning if he and his family were from a cultural community that particularly emphasized family bonds so in other words in order to really develop a full understanding of mental health we must develop an understanding of individuals in context so this discussion um illustrates one of the key challenges of a culturally informed approach to mental health and in teaching this field to undergraduate students I have the privileged opportunity of working with them as they wrestle with these very types of challenges the path that I often see my students take is one of initial excitement as they engage with the idea that culture matters and often tend to appreciate the in-depth discussion of a topic that tends to be lacking or minimized in teachings about mental health this initial excitement though then turns often to some level of feeling overwhelmed as they begin to really grapple with the complexity of culture and begin to recognize the number of different factors to consider and questions to pose in helping my students to navigate these questions it has helped me to distill my own thinking about this field and how best to teach it I recognize that dealing with cultural matters can indeed be daunting and I also have the concern that feeling overwhelmed can sometimes lead to the avoidance of new ideas or experiences so with that in mind I propose two related suggestions that I believe can help people to engage with a culturally informed form Med approach to mental health rather than avoid it and these are to take a stance of informed curiosity and to ask different questions and ask questions differently so a stance of informed curiosity stands in contrast to some models of culture and mental health which suggest that one needs to be an expert in a particular cultural group in order to work with or understand the experiences of a member of that group for example many articles and book chapters talk about working with Asian clients or working with Hispanic clients although well intentioned that type of approach can inadvertently lead to the perpetuation of cultural stereotypes there's the Assumption at least at some level that members of a given group share values beliefs and experiences simply based on group membership though that can be the case it is certainly not necessarily true a clinician working from such a perspective may have easily assumed that Patricia Lee's level of dependence upon her family was encouraged or supported I also always found the more expertise-based approach problematic from a personal perspective as the daughter of a Chinese Canadian father and Anglo Canadian mother born and raised in bilingual Montreal with English as a first language there is unlikely to ever be a book chapter devoted to working with people of my particular culture cultural A Stance of Informed Curiosity makeup I think my own lived experience as a person of mixed cultural heritage informs my thinking about culture and how it plays a role in mental health and my experience also plays a role in my emphasis on a stance of informed curiosity rather than expertise I believe that such a stance promotes an attitude of humility and really encourages one to develop hypotheses and questions rather than make assumptions so one of the key ways that I work to promote a stance of informed curiosity among my students is by helping them to ask different questions and ask questions differently about mental health my own passion for the area of culture and mental health really began when I took a course an introductory course in medical anthropology in the second year of my undergraduate studies this course proved to be a key turning point in my academic path and looking back I think that one of its last effects was that it forced me to ask different questions about health and mental health questions to which I had not previously been exposed in my psychology courses working with a different discipline with its own set of theories and Frameworks also instilled in me the value of asking questions differently in many ways that course forced me to re-evaluate ideas I had previously taken for granted and that type of experience is often at the heart of cultural discussions about m mental Cultural Differences health the interdisciplinary nature of my early experience also corresponds to the interdisciplinarity that often characterizes discussions of culture and mental health thinking back to that image of the complex set of puzzle pieces a cultural approach to mental health truly benefits the from the integration of different perspectives each offering their own sets and types of questions now taking a stance of inform curiosity by no means minimizes the importance of developing in-depth knowledge in cultural matters rather the emphasis is on how to engage with this knowledge with a focus really on gaining awareness of the wide range of cultural differences that exist in Human Experience so as to be equipped to pose informed questions we know from cultural psychology research that there are significant cultural differences across the Spectrum of psychological domains including in cognitive and perceptual processes regarded as basic and therefore often assumed to be Universal for example we know that there are significant cultural differences in how people perceive the world such as to what extent they focus on background information when examining a photograph versus focusing on the photograph's central figure we also know that there are significant differences in how people think about the world such as to what extent they use contextual information when grouping different objects together these types of differences suggest that some of you may be viewing the slide behind me as a single holistic image whereas others may be more focused on specific constituent Parts more directly relevant to mental health we also know that cultural context plays a profound role in what types of emotions people most value and want to feel and how and with whom they express those emotions in some cultural contexts Cultural Context feeling proud is closely associated with generally feeling good whereas in other contexts the individual Focus that Pride entails makes it less desirable and there is a large literature demonstrating important cultural differences in how people Define the very notion of the self for some the self is truly an autonomous and individual entity whereas for others their ideas about the self include close interpersonal connections so cultural differences in domains such as these should directly inform our thinking about mental health and illness if culture plays such a profound role when things are going right it must surely play a similar role when things go arai how people experience profound shifts in their emotional life is deeply influenced by their values and beliefs about emotions and when and with whom it is appropriate to share their emotions in some cultural contexts physical symptoms of major depression tend to be more strongly emphasized compared to psychological symptoms and this may be linked to cultural differences in values about emotion so at this point I have likely Rewards convinced you of a number of the challenges of taking a culturally informed approach to mental health and I have offered some suggestions about how to engage with these challenges but hopefully I have also at least planted the seed that there are rewards to be had as well not surprisingly I personally believe that the rewards at the level of research clinical care and teaching about mental health are numerous and profound and ultimately far outweigh the challenges at a research level in order to build a comprehensive understanding of mental illness we must integrate cultural factors as suggested by the quote near the start of my talk we neglect culture at our own Peril if what it means to be sad and to share this sadness with others is profoundly impacted by our values and beliefs about emotion then we will never develop a complete understanding of severe sadness or depression without a cultural approach from a clinical perspective integrating cultural factors into our assessment of a person's distress is crucial to really understand their experience and can contribute to improved treatment we know from research that treatments that integrate a person's own cultural beliefs about the causes of their distress are more effective than treatments that do not how best to integrate cultural factors into treatment is an area of ongoing investigation and debate but begins with the argument that culture makes a difference and thirdly at the level of learning about mental health for students and the general public more broadly the rewards of a culturally informed approach include greater open-mindedness increased understanding of experiences different from one's own and less stereotyping this approach also offers New Perspectives about ideas already learned about mental health these are all rewards that my own students have specifically talked about when discussing the benefits of learning about this field so my hope then for today is that you walk away with some of this increased open-mindedness curiosity and an interest in asking different questions about mental health all rooted in the central premise that culture matters thank you",
            "In the mid-'90s, the CDC and Kaiser Permanente discovered an exposure that dramatically increased the risk for seven out of 10 of the leading causes of death in the United States. In high doses, it affects brain development, the immune system, hormonal systems, and even the way our DNA is read and transcribed. Folks who are exposed in very high doses have triple the lifetime risk of heart disease and lung cancer and a 20-year difference in life expectancy. And yet, doctors today are not trained in routine screening or treatment. Now, the exposure I'm talking about is not a pesticide or a packaging chemical. It's childhood trauma. Okay. What kind of trauma am I talking about here? I'm not talking about failing a test or losing a basketball game. I am talking about threats that are so severe or pervasive that they literally get under our skin and change our physiology: things like abuse or neglect, or growing up with a parent who struggles with mental illness or substance dependence. Now, for a long time, I viewed these things in the way I was trained to view them, either as a social problem -- refer to social services -- or as a mental health problem -- refer to mental health services. And then something happened to make me rethink my entire approach. When I finished my residency, I wanted to go someplace where I felt really needed, someplace where I could make a difference. So I came to work for California Pacific Medical Center, one of the best private hospitals in Northern California, and together, we opened a clinic in Bayview-Hunters Point, one of the poorest, most underserved neighborhoods in San Francisco. Now, prior to that point, there had been only one pediatrician in all of Bayview to serve more than 10,000 children, so we hung a shingle, and we were able to provide top-quality care regardless of ability to pay. It was so cool. We targeted the typical health disparities: access to care, immunization rates, asthma hospitalization rates, and we hit all of our numbers. We felt very proud of ourselves. But then I started noticing a disturbing trend. A lot of kids were being referred to me for ADHD, or Attention Deficit Hyperactivity Disorder, but when I actually did a thorough history and physical, what I found was that for most of my patients, I couldn't make a diagnosis of ADHD. Most of the kids I was seeing had experienced such severe trauma that it felt like something else was going on. Somehow I was missing something important. Now, before I did my residency, I did a master's degree in public health, and one of the things that they teach you in public health school is that if you're a doctor and you see 100 kids that all drink from the same well, and 98 of them develop diarrhea, you can go ahead and write that prescription for dose after dose after dose of antibiotics, or you can walk over and say, \"What the hell is in this well?\" So I began reading everything that I could get my hands on about how exposure to adversity affects the developing brains and bodies of children. And then one day, my colleague walked into my office, and he said, \"Dr. Burke, have you seen this?\" In his hand was a copy of a research study called the Adverse Childhood Experiences Study. That day changed my clinical practice and ultimately my career. The Adverse Childhood Experiences Study is something that everybody needs to know about. It was done by Dr. Vince Felitti at Kaiser and Dr. Bob Anda at the CDC, and together, they asked 17,500 adults about their history of exposure to what they called \"adverse childhood experiences,\" or ACEs. Those include physical, emotional, or sexual abuse; physical or emotional neglect; parental mental illness, substance dependence, incarceration; parental separation or divorce; or domestic violence. For every yes, you would get a point on your ACE score. And then what they did was they correlated these ACE scores against health outcomes. What they found was striking. Two things: Number one, ACEs are incredibly common. Sixty-seven percent of the population had at least one ACE, and 12.6 percent, one in eight, had four or more ACEs. The second thing that they found was that there was a dose-response relationship between ACEs and health outcomes: the higher your ACE score, the worse your health outcomes. For a person with an ACE score of four or more, their relative risk of chronic obstructive pulmonary disease was two and a half times that of someone with an ACE score of zero. For hepatitis, it was also two and a half times. For depression, it was four and a half times. For suicidality, it was 12 times. A person with an ACE score of seven or more had triple the lifetime risk of lung cancer and three and a half times the risk of ischemic heart disease, the number one killer in the United States of America. Well, of course this makes sense. Some people looked at this data and they said, \"Come on. You have a rough childhood, you're more likely to drink and smoke and do all these things that are going to ruin your health. This isn't science. This is just bad behavior.\" It turns out this is exactly where the science comes in. We now understand better than we ever have before how exposure to early adversity affects the developing brains and bodies of children. It affects areas like the nucleus accumbens, the pleasure and reward center of the brain that is implicated in substance dependence. It inhibits the prefrontal cortex, which is necessary for impulse control and executive function, a critical area for learning. And on MRI scans, we see measurable differences in the amygdala, the brain's fear response center. So there are real neurologic reasons why folks exposed to high doses of adversity are more likely to engage in high-risk behavior, and that's important to know. But it turns out that even if you don't engage in any high-risk behavior, you're still more likely to develop heart disease or cancer. The reason for this has to do with the hypothalamic–pituitary–adrenal axis, the brain's and body's stress response system that governs our fight-or-flight response. How does it work? Well, imagine you're walking in the forest and you see a bear. Immediately, your hypothalamus sends a signal to your pituitary, which sends a signal to your adrenal gland that says, \"Release stress hormones! Adrenaline! Cortisol!\" And so your heart starts to pound, Your pupils dilate, your airways open up, and you are ready to either fight that bear or run from the bear. And that is wonderful if you're in a forest and there's a bear. (Laughter) But the problem is what happens when the bear comes home every night, and this system is activated over and over and over again, and it goes from being adaptive, or life-saving, to maladaptive, or health-damaging. Children are especially sensitive to this repeated stress activation, because their brains and bodies are just developing. High doses of adversity not only affect brain structure and function, they affect the developing immune system, developing hormonal systems, and even the way our DNA is read and transcribed. So for me, this information threw my old training out the window, because when we understand the mechanism of a disease, when we know not only which pathways are disrupted, but how, then as doctors, it is our job to use this science for prevention and treatment. That's what we do. So in San Francisco, we created the Center for Youth Wellness to prevent, screen and heal the impacts of ACEs and toxic stress. We started simply with routine screening of every one of our kids at their regular physical, because I know that if my patient has an ACE score of 4, she's two and a half times as likely to develop hepatitis or COPD, she's four and half times as likely to become depressed, and she's 12 times as likely to attempt to take her own life as my patient with zero ACEs. I know that when she's in my exam room. For our patients who do screen positive, we have a multidisciplinary treatment team that works to reduce the dose of adversity and treat symptoms using best practices, including home visits, care coordination, mental health care, nutrition, holistic interventions, and yes, medication when necessary. But we also educate parents about the impacts of ACEs and toxic stress the same way you would for covering electrical outlets, or lead poisoning, and we tailor the care of our asthmatics and our diabetics in a way that recognizes that they may need more aggressive treatment, given the changes to their hormonal and immune systems. So the other thing that happens when you understand this science is that you want to shout it from the rooftops, because this isn't just an issue for kids in Bayview. I figured the minute that everybody else heard about this, it would be routine screening, multi-disciplinary treatment teams, and it would be a race to the most effective clinical treatment protocols. Yeah. That did not happen. And that was a huge learning for me. What I had thought of as simply best clinical practice I now understand to be a movement. In the words of Dr. Robert Block, the former President of the American Academy of Pediatrics, \"Adverse childhood experiences are the single greatest unaddressed public health threat facing our nation today.\" And for a lot of people, that's a terrifying prospect. The scope and scale of the problem seems so large that it feels overwhelming to think about how we might approach it. But for me, that's actually where the hopes lies, because when we have the right framework, when we recognize this to be a public health crisis, then we can begin to use the right tool kit to come up with solutions. From tobacco to lead poisoning to HIV/AIDS, the United States actually has
