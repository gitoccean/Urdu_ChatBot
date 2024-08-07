import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something in Urdu!")
        audio = r.listen(source)
        st.write("Recording complete!")
    return audio

def transcribe_audio(audio):
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio, language='ur')
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

OPEN_AI_API_KEY = ""

def generate_response(text):
    try:
        chat = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, temperature=0.2)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "As an AI assistant, your expertise lies in accurately translating Voice content into Urdu Voice languages"
                ),
                ("human", "{input}"),
            ]
        )
        chain = prompt | chat
        response = chain.invoke(
            {
                "input": text,
            }
        )
        return response.content
    except Exception as e:
        st.write(f"Error: {e}")

def text_to_speech(text, lang='ur'):
    tts = gTTS(text=text, lang=lang)
    audio_file_path = "response.mp3"
    tts.save(audio_file_path)
    return audio_file_path

def play_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    play(audio)

def main():
    st.title("Urdu Voice Assistant")
    if st.button("Record Voice"):
        audio = record_audio()
        text = transcribe_audio(audio)
        st.write("Transcribed Text:", text)

        if text:
            response = generate_response(text)
            st.write("Response:", response)
            audio_file = text_to_speech(response)
            st.audio(audio_file, format="audio/mp3")
            # play_audio(audio_file)

if __name__ == "__main__":
    main()
