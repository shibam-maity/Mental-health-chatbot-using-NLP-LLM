import streamlit as st
import chromadb
import os
import time
import json
from typing import Dict, Tuple, List
from transformers import pipeline
from dotenv import load_dotenv
from groq import Groq
from langdetect import detect
from deep_translator import GoogleTranslator  # Replace googletrans with deep-translator
# Update Eleven Labs import to use the correct API
import elevenlabs
from elevenlabs.client import ElevenLabs

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    GROQ_API_KEY = "your_api_key_here"  # Replace with your actual API key for testing

# Configure Eleven Labs API
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
if ELEVEN_LABS_API_KEY:
    # Set the API key using the proper method for this elevenlabs version
    elevenlabs.api_key = ELEVEN_LABS_API_KEY
    eleven_labs_available = True
else:
    eleven_labs_available = False

# Configure Groq AI
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama3-70b-8192"  # You can change this to other Groq models as needed

# Initialize ChromaDB for RAG
client = chromadb.PersistentClient(path="./mental_health_memory")
collection = client.get_or_create_collection(name="chat_history")

# Function to analyze sentiment using LLM instead of sentiment_pipeline
# The LLM will be prompted to output a JSON with emotion scores

def analyze_sentiment(text):
    prompt = (
        "Analyze the following user's message for emotions. "
        "Return a JSON object with the keys: sadness, joy, love, anger, fear, surprise and their corresponding scores (0 to 1, sum to 1). "
        "Also, provide the most likely detected emotion as 'Detected Emotion'.\n"
        f"User message: {text}\n"
        "Response format:\n"
        "Detected Emotion: <emotion>\n"
        "{\n'sadness':..., 'joy':..., 'love':..., 'anger':..., 'fear':..., 'surprise':...}\n"
    )
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    content = response.choices[0].message.content
    # Parse detected emotion and scores from LLM output
    import re
    detected_emotion = None
    emotions = {}
    # Extract Detected Emotion
    match = re.search(r"Detected Emotion:\s*(\w+)", content)
    if match:
        detected_emotion = match.group(1).lower()
    # Extract JSON block
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            emotions = json.loads(match.group(0).replace("'", '"'))
        except Exception:
            emotions = {}
    return detected_emotion, emotions

# Function to store chat history in ChromaDB
def store_chat(user_input, bot_response):
    collection.add(
        documents=[user_input, bot_response],
        metadatas=[{"role": "user"}, {"role": "bot"}],
        ids=[str(len(collection.get())) + "_user", str(len(collection.get())) + "_bot"]
    )

# Function to retrieve relevant past messages (RAG)
def retrieve_context():
    history = collection.get()
    if len(history["documents"]) > 3:
        return history["documents"][-3:]
    return history["documents"]

# Define AI Agents
class AgentCoordinator:
    def __init__(self, therapist, emotional_support, resource_recommender):
        self.therapist = therapist
        self.emotional_support = emotional_support
        self.resource_recommender = resource_recommender
        self.conversation_state = {
            "session_emotions": [],
            "topics_discussed": set(),
            "crisis_detected": False,
            "response_styles_used": set()
        }
        self.supported_languages = ['en', 'es', 'fr', 'de', 'hi', 'ar', 'zh-cn', 'ja']

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return lang if lang in self.supported_languages else 'en'
        except:
            return 'en'

    def translate_response(self, text, target_lang):
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            return f"[Translation Error: {str(e)}]"

    def update_conversation_state(self, user_input, emotion, emotion_scores):
        self.conversation_state["session_emotions"].append((emotion, max(emotion_scores.values())))
        crisis_keywords = ["suicide", "kill myself", "end my life", "harm myself", "die"]
        if any(keyword in user_input.lower() for keyword in crisis_keywords):  
            self.conversation_state["crisis_detected"] = True

        mental_health_topics = ["anxiety", "depression", "stress", "trauma", "sleep",
                               "relationship", "work", "family", "addiction", "grief"]
        for topic in mental_health_topics:
            if topic in user_input.lower():
                self.conversation_state["topics_discussed"].add(topic)

    def determine_response_style(self, user_input, emotion):
        input_length = len(user_input.split())
        if input_length <= 5:
            return "brief"
        if self.conversation_state["crisis_detected"]:
            return "crisis_support"
        if emotion in ["sadness", "fear"]:
            return "compassionate"
        if emotion == "anger":
            return "de-escalation"
        if emotion == "joy":
            return "affirming"
        return "balanced"

    def coordinate_response(self, user_input, context):
        # Detect language first
        user_lang = self.detect_language(user_input)
        
        emotion, emotion_scores = self.emotional_support.analyze_emotional_state(user_input)
        self.update_conversation_state(user_input, emotion, emotion_scores)
        response_style = self.determine_response_style(user_input, emotion)
        self.conversation_state["response_styles_used"].add(response_style)

        is_greeting = any(greeting in user_input.lower() for greeting in ["hi", "hello", "hey", "greetings"])
        is_question = "?" in user_input
        input_complexity = len(user_input.split())

        response_plan = {
            "main_agent": "therapist",
            "activate_emotional_support": False,
            "activate_resources": False,
            "response_style": response_style,
            "max_response_length": 150 if input_complexity < 10 else 300
        }

        if self.conversation_state["crisis_detected"]:
            response_plan["activate_resources"] = True
            response_plan["response_style"] = "crisis_support"

        if emotion in ["sadness", "fear", "anger"] and emotion_scores[emotion] > 0.6:
            response_plan["activate_emotional_support"] = True

        help_keywords = ["help", "resource", "support", "where", "how to", "recommend", "suggestion"]
        if any(keyword in user_input.lower() for keyword in help_keywords) or is_question:
            response_plan["activate_resources"] = True

        # Get responses in English first
        main_response = self.therapist.generate_response(
            user_input,
            context,
            emotion,
            response_style=response_plan["response_style"],
            max_length=response_plan["max_response_length"]
        )

        coping_strategies = ""
        resources = ""

        if response_plan["activate_emotional_support"]:
            coping_strategies = self.emotional_support.generate_coping_strategies(
                emotion,
                emotion_scores[emotion],
                topics=list(self.conversation_state["topics_discussed"])
            )

        if response_plan["activate_resources"]:
            resources = self.resource_recommender.recommend_resources(
                emotion,
                user_input,
                crisis_mode=self.conversation_state["crisis_detected"]
            )

        # Translate responses if needed
        if user_lang != 'en':
            main_response = self.translate_response(main_response, user_lang)
            if coping_strategies:
                coping_strategies = self.translate_response(coping_strategies, user_lang)
            if resources:
                resources = self.translate_response(resources, user_lang)

        return {
            "main_response": main_response,
            "coping_strategies": coping_strategies,
            "resources": resources,
            "emotion": emotion,
            "show_coping": bool(coping_strategies),
            "show_resources": bool(resources),
            "response_style": response_plan["response_style"],
            "language": user_lang
        }

class TherapistAgent:
    def __init__(self, groq_client, model_name):
        self.groq_client = groq_client
        self.model_name = model_name

    def generate_response(self, user_input: str, context: List[str], emotion: str,
                         response_style: str = "balanced", max_length: int = 300) -> str:
        style_prompts = {
            "brief": "Keep your response concise and to the point, under 100 words.",
            "compassionate": "Respond with warmth and empathy, acknowledging the user's feelings.",
            "de-escalation": "Use a calm tone to help reduce tension and anger.",
            "crisis_support": "Provide immediate support and safety-focused guidance.",
            "affirming": "Celebrate positive emotions and reinforce healthy behaviors.",
            "balanced": "Provide a balanced therapeutic response with moderate length."
        }

        style_instruction = style_prompts.get(response_style, style_prompts["balanced"])
        input_length = len(user_input.split())
        is_greeting = any(greeting in user_input.lower() for greeting in ["hi", "hello", "hey", "greetings"])

        if is_greeting and input_length <= 3 and not context:
            messages = [
                {"role": "system", "content": "You are a friendly mental health chatbot. Keep responses brief and natural."},
                {"role": "user", "content": f"The user said: '{user_input}'. Respond with a brief, friendly greeting under 30 words."}
            ]
        else:
            messages = [
                {"role": "system", 
                 "content": f"You are a multilingual therapeutic AI agent. Respond in English but be prepared to handle inputs in various languages."
                },
                {"role": "user", "content": f"""As a therapeutic AI agent, provide an appropriate response.
                User's Emotion: {emotion}
                Previous Context: {context}
                User Input: {user_input}
                Response Style: {response_style}
                Maximum Length: {max_length} words
                Avoid starting with 'I'm sorry to hear' or similar phrases unless truly appropriate."""}
            ]

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an issue generating a response. Error: {str(e)}"

class EmotionalSupportAgent:
    def __init__(self, groq_client, model_name):
        self.groq_client = groq_client
        self.model_name = model_name

    def analyze_emotional_state(self, text: str):
        return analyze_sentiment(text)

    def generate_coping_strategies(self, emotion: str, intensity: float, topics: List[str] = None):
        prompt = (
            f"Suggest coping strategies for someone feeling {emotion} with intensity {intensity}. "
            f"Topics: {topics if topics else 'general'}"
        )
        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

class ResourceRecommenderAgent:
    def __init__(self, groq_client, model_name):
        self.groq_client = groq_client
        self.model_name = model_name

    def recommend_resources(self, emotion: str, user_input: str, crisis_mode: bool = False) -> str:
        priority = "high priority Crisis resources" if crisis_mode else "helpful resources"

        messages = [
            {"role": "system", "content": "You are an AI specialized in recommending mental health resources. Be concise and practical."},
            {"role": "user", "content": f"""Based on the user's emotion ({emotion}) and input: '{user_input}',
                recommend 2-3 {priority}. Keep your response under 150 words and focus on specific, actionable recommendations."""}
        ]

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an issue generating resource recommendations. Error: {str(e)}"

# Initialize Agents
therapist = TherapistAgent(groq_client, MODEL_NAME)
emotional_support = EmotionalSupportAgent(groq_client, MODEL_NAME)
resource_recommender = ResourceRecommenderAgent(groq_client, MODEL_NAME)
coordinator = AgentCoordinator(therapist, emotional_support, resource_recommender)

def get_multi_agent_response(user_input: str) -> Dict[str, str]:
    past_context = retrieve_context()
    return coordinator.coordinate_response(user_input, past_context)

# Streamlit UI
st.title("\U0001F9E0 Mental Health Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

user_input = st.chat_input("Ask me anything about mental health...")

if user_input:
    st.chat_message("user").write(user_input)
    with st.spinner("Thinking..."):
        responses = get_multi_agent_response(user_input)

    with st.chat_message("assistant"):
        st.write(responses["main_response"])

        if responses["show_coping"] and responses["coping_strategies"]:
            with st.expander("📋 Coping Strategies"):
                st.write(responses["coping_strategies"])

        if responses["show_resources"] and responses["resources"]:
            with st.expander("🔍 Helpful Resources"):
                st.write(responses["resources"])

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", responses["main_response"]))
    store_chat(user_input, responses["main_response"])

# Add to sidebar
with st.sidebar:
    st.header("Settings")
    default_lang = st.selectbox(
        "Preferred Response Language",
        ["Auto-detect", "English", "Spanish", "French", "German", "Hindi", "Arabic", "Chinese", "Japanese"],
        index=0
    )
    
    # Eleven Labs section
    st.markdown("## Voice Assistant")
    
    if not eleven_labs_available:
        st.warning("Eleven Labs API key not found. You need to add your API key to use text-to-speech.")
        st.markdown("Get your API key at [Eleven Labs](https://elevenlabs.io/)")
        user_api_key = st.text_input("Enter your Eleven Labs API key:", type="password")
        if user_api_key:
            elevenlabs.api_key = user_api_key
            eleven_labs_available = True
            st.success("API key set successfully!")
    
    if eleven_labs_available:
        elevenlabs_html = """
        <div style="width: 100%; margin: 0 auto;">
            <elevenlabs-convai agent-id="jw3AbtarA0MWqlpksUb3"></elevenlabs-convai>
            <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
        </div>
        """
        st.components.v1.html(elevenlabs_html, height=150)  # Increased height for better visibility
    
    st.markdown("---")
    st.markdown("*Note: This chatbot is for informational purposes only and should not replace professional mental health advice.*")

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    collection.delete(ids=collection.get()["ids"])
    st.success("Chat cleared! Refreshing...")
    st.rerun()

if st.button("Sentiment Analysis"):
    chat_text = " ".join([msg for _, msg in st.session_state.chat_history])
    if chat_text:
        detected_emotion, emotion_scores = analyze_sentiment(chat_text)
        st.subheader("Sentiment Analysis Result")
        st.write(f"Detected Emotion: **{detected_emotion}**")
        st.json(emotion_scores)
    else:
        st.warning("No chat history found for analysis.")